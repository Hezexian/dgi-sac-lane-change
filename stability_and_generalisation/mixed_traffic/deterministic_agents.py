"""Deterministic baselines on historical SAC observations and uniform batches."""
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from algorithms.sac_base.models.critic import ValueNetwork, weights_init_
from core import (LegacyBatch, historical_config, isolated_rng, rng_state,
                         restore_rng, seed_everything)
from utils import soft_update


# Observation scaling correction (2026-10-01): with raw highway-env kinematics (ego x up to ~1300 m) the
# deterministic tanh actor was born saturated (99.96% |a|>0.99, zero actor gradient). Following
# DDPG (Lillicrap et al. 2016): fixed input scaling for heterogeneous physical units, and final
# layers initialised U(-3e-3, 3e-3). Scales = 1/p99 |obs| per (ego/other, feature) measured on the
# source run LC-N2-261001-017 150K replay; applied identically to actor and critic inputs.
# ego-centric-fixed-scale: inputs are already ego-centric and fixed-scale normalised by observation (shared by all
# methods), so the earlier private 1/p99 scaling is neutralised to avoid scaling twice. The DDPG final-layer
# U(-3e-3, 3e-3) initialisation is retained.
_EGO = [1.]*6
_OTHER = [1.]*6
OBS_SCALE = torch.tensor(_EGO + _OTHER*7, dtype=torch.float32)


def final_init(layer):
    nn.init.uniform_(layer.weight, -3e-3, 3e-3)
    nn.init.uniform_(layer.bias, -3e-3, 3e-3)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(48, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 2)
        self.apply(weights_init_)
        final_init(self.linear3)
        self.register_buffer('obs_scale', OBS_SCALE.clone())

    def forward(self, state):
        state = state/self.obs_scale
        return torch.tanh(self.linear3(F.relu(self.linear2(F.relu(self.linear1(state))))))


class Critic(nn.Module):
    def __init__(self, twin):
        super().__init__()
        self.heads = nn.ModuleList([ValueNetwork(50, 256) for _ in range(2 if twin else 1)])
        for head in self.heads:
            final_init(head.linear3)
        self.register_buffer('obs_scale', torch.cat([OBS_SCALE, torch.ones(2)]))

    def forward(self, state, action):
        inputs = torch.cat((state, action), dim=1)/self.obs_scale
        return tuple(head(inputs) for head in self.heads)


class TDAgent:
    def __init__(self, variant, seed, device='cpu', config=None, action_space=None, threads=1):
        if variant not in {'ddpg', 'td3'}:
            raise ValueError('Expected ddpg or td3')
        if action_space is not None and (tuple(action_space.shape) != (2,) or
                not np.all(action_space.low == -1.) or not np.all(action_space.high == 1.)):
            raise ValueError('Deterministic baselines require two actions in [-1, 1]')
        seed_everything(seed, threads)
        self.variant, self.seed, self.device = variant, seed, torch.device(device)
        self.config = copy.deepcopy(config if config is not None else historical_config('sac'))
        self.config['run'].update(seed=seed, cuda=self.device.type == 'cuda')
        self.legacy = self.capacity_match = None
        self.gamma, self.tau = .99, .005
        self.policy = Actor().to(self.device)
        self.critic = Critic(variant == 'td3').to(self.device)
        self.policy_target = copy.deepcopy(self.policy)
        self.critic_target = copy.deepcopy(self.critic)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.exploration_rng = np.random.default_rng(seed)
        self.delay_counter = 0
        self.counts = dict(actor_updates=0, critic_optimizer_updates=0, target_updates=0,
                           actor_samples=0, critic_samples=0, alpha_updates=0)
        self.last_actor = dict(actor_loss=None, actor_grad_norm=None, action_saturation_fraction=None)

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.as_tensor(np.asarray(state).reshape(1, -1), dtype=torch.float32,
                                    device=self.device)
            action = self.policy(state).cpu().numpy()[0]
        if not evaluate:
            action = np.clip(action + self.exploration_rng.normal(0., .1, action.shape), -1., 1.)
        return action

    def target_value(self, next_state):
        action = self.policy_target(next_state)
        if self.variant == 'td3':
            noise = (torch.randn_like(action)*.2).clamp(-.5, .5)
            action = (action + noise).clamp(-1., 1.)
        return torch.stack(self.critic_target(next_state, action)).amin(dim=0)

    def update(self, memory, batch_size, step, episode_step, budget, fixed_indices=None):
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        s, actions, rewards, ns, mask = (tensor(x) for x in
                LegacyBatch(memory, fixed_indices).sample(batch_size))
        with torch.no_grad():
            y = rewards.reshape(-1, 1) + mask.reshape(-1, 1)*self.gamma*self.target_value(ns)
        critic_loss = sum(F.mse_loss(q, y) for q in self.critic(s, actions))
        if not torch.isfinite(critic_loss):
            raise FloatingPointError('Nonfinite deterministic critic loss')
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self.counts['critic_optimizer_updates'] += 1
        self.counts['critic_samples'] += batch_size
        self.delay_counter += 1
        if self.variant == 'ddpg' or self.delay_counter % 2 == 0:
            policy_actions = self.policy(s)
            actor_loss = -self.critic(s, policy_actions)[0].mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError('Nonfinite deterministic actor loss')
            self.policy_optim.zero_grad()
            actor_loss.backward()
            grad_norm = float(torch.sqrt(sum(p.grad.detach().square().sum()
                              for p in self.policy.parameters() if p.grad is not None)))
            saturation = float((policy_actions.detach().abs() > .99).float().mean())
            self.policy_optim.step()
            soft_update(self.policy_target, self.policy, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
            self.counts['actor_updates'] += 1
            self.counts['actor_samples'] += batch_size
            # One target sweep updates both actor and critic target networks.
            self.counts['target_updates'] += 1
            self.last_actor = dict(actor_loss=float(actor_loss.detach()), actor_grad_norm=grad_norm,
                                   action_saturation_fraction=saturation)
        return {'critic_loss': float(critic_loss.detach()), **self.last_actor}

    def state_dict(self):
        return {'variant': self.variant, 'seed': self.seed, 'config': copy.deepcopy(self.config),
                'counts': self.counts.copy(), 'capacity_match': None,
                'policy_state_dict': self.policy.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'policy_target_state_dict': self.policy_target.state_dict(),
                'critic_target_state_dict': self.critic_target.state_dict(),
                'policy_optimizer_state_dict': self.policy_optim.state_dict(),
                'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                'exploration_rng': copy.deepcopy(self.exploration_rng.bit_generator.state),
                'delay_counter': self.delay_counter, 'last_actor': self.last_actor.copy(),
                # LegacyBatch and smoothing use global Python/Torch streams. Runner
                # also stores these; including them allows standalone exact resume.
                'rng': rng_state()}

    def load_state_dict(self, d):
        if d['variant'] != self.variant:
            raise ValueError('Checkpoint algorithm mismatch')
        for key, obj in [('policy_state_dict', self.policy), ('critic_state_dict', self.critic),
                         ('policy_target_state_dict', self.policy_target),
                         ('critic_target_state_dict', self.critic_target),
                         ('policy_optimizer_state_dict', self.policy_optim),
                         ('critic_optimizer_state_dict', self.critic_optim)]:
            if key in d:
                obj.load_state_dict(d[key])
        self.seed, self.config = d['seed'], copy.deepcopy(d['config'])
        self.counts, self.last_actor = d['counts'].copy(), d['last_actor'].copy()
        self.delay_counter = d['delay_counter']
        self.exploration_rng.bit_generator.state = copy.deepcopy(d['exploration_rng'])
        restore_rng(d['rng'])

    def diagnostics(self, memory, step):
        """Uniform probe, isolated all RNG, no training/replay mutations."""
        if memory.size == 0:
            return {}
        with isolated_rng(), torch.no_grad():
            r = np.random.default_rng(self.seed + int(step) + 80000000)
            ids = r.choice(memory.size, min(4096, memory.size), replace=False)
            errors = []
            tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
            for pos in range(0, len(ids), 256):
                ix = ids[pos:pos+256]
                s, ns = tensor(memory.obs[ix]).flatten(1), tensor(memory.next_obs[ix]).flatten(1)
                y = tensor(memory.reward[ix]).reshape(-1, 1) + self.gamma*self.target_value(ns)*(
                        1. - tensor(memory.terminal[ix]).reshape(-1, 1))
                qs = self.critic(s, tensor(memory.action[ix]))
                error = torch.stack([(q-y).abs() for q in qs]).amax(dim=0)
                errors.extend(error.flatten().cpu().tolist())
            e = np.asarray(errors)
            return {'uniform_td_probe_n': len(e), 'current_td_floor_fraction': float(np.mean(e <= 1.)),
                    'current_td_quantiles': np.quantile(e, [0, .25, .5, .75, .9, .99, 1]).tolist()}

    def parameter_counts(self):
        return {'actor': sum(p.numel() for p in self.policy.parameters()),
                'critic': sum(p.numel() for p in self.critic.parameters()),
                'target_critic': sum(p.numel() for p in self.critic_target.parameters())}
