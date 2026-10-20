"""SAC agent wrapper: historical networks and SAC update order remain authoritative."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import contextlib
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from algorithms import load_algorithm
from configs.config_loader import build_train_args, load_config
from utils import soft_update
from stabilizers import validate_stabilizer, has_qclip, has_pbrs, q_upper_bound, clip_target

LEGACY = {'sac', 'dgi', 'capacity'}


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


@contextlib.contextmanager
def isolated_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def historical_config(variant):
    filename = 'baseline.yaml' if variant in {'sac', 'capacity'} else 'sac_gcn.yaml'
    _, config = load_config(Path(__file__).parent / 'configs' / filename)
    return config


class LegacyBatch:
    """Original random.sample draw; same batch feeds both historical optimizers."""
    def __init__(self, memory, indices=None):
        self.memory = memory
        self.indices = indices

    def sample(self, batch_size):
        m = self.memory
        ids = self.indices if self.indices is not None else random.sample(range(m.size), batch_size)
        self.sampled_indices = ids
        return (m.obs[ids].reshape(batch_size, -1), m.action[ids], m.reward[ids],
                m.next_obs[ids].reshape(batch_size, -1), 1.0 - m.terminal[ids])


class Agent:
    def __init__(self, variant, seed, device='cpu', config=None, action_space=None, threads=1,
                 priority_mode='floor', stabilizer='none'):
        validate_stabilizer(variant, stabilizer)
        self.stabilizer = stabilizer
        seed_everything(seed, threads)
        self.variant, self.seed, self.priority_mode = variant, seed, priority_mode
        self.config = copy.deepcopy(config if config is not None else historical_config(variant))
        self.config['run']['seed'] = seed
        self.config['run']['cuda'] = device == 'cuda'
        if action_space is None:
            from gymnasium.spaces import Box
            action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        args = build_train_args(self.config, 'experiment-config.yaml')
        if args.automatic_entropy_tuning or args.alpha != .025:
            raise ValueError('SAC baseline requires fixed alpha .025')
        self.legacy = load_algorithm(self.config['algorithm']['name']).create_agent(
            args.obs_flat_dim, action_space, args)
        if has_qclip(stabilizer):
            if action_space.shape != (2,) or not (np.all(action_space.low == -1) and np.all(action_space.high == 1)):
                raise ValueError('Q bound requires action support [-1,1]^2')
            self.legacy.q_max = q_upper_bound(self.config, stabilizer)
        self.device = self.legacy.device
        self.capacity_match = None
        if variant == 'capacity':
            from algorithms.sac_base.models.policy import GaussianPolicy
            with isolated_rng():
                graph_args = build_train_args(historical_config('dgi'), 'graph-config.yaml')
                graph_args.cuda = device == 'cuda'
                graph_agent = load_algorithm('sac_gcn').create_agent(args.obs_flat_dim, action_space, graph_args)
                target = sum(p.numel() for p in graph_agent.policy.parameters())
                del graph_agent
            inp, out = args.obs_flat_dim, action_space.shape[0]
            count = lambda w: inp*w+w + w*w+w + 2*(w*out+out)
            width = min(range(256, 2049), key=lambda w: abs(count(w)-target))
            self.legacy.policy = GaussianPolicy(inp, out, width, action_space).to(self.device)
            self.legacy.policy_optim = torch.optim.Adam(self.legacy.policy.parameters(), lr=args.lr)
            self.capacity_match = {'width': width, 'target': target, 'actual': count(width),
                                   'relative_difference': abs(count(width)-target)/target}
            assert self.capacity_match['relative_difference'] < .01
        self.counts = dict(actor_updates=0, critic_optimizer_updates=0, target_updates=0,
                           actor_samples=0, critic_samples=0, alpha_updates=0)

    @property
    def policy(self):
        return self.legacy.policy

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            return self.legacy.select_action(np.asarray(state).reshape(-1), evaluate=evaluate)

    def update(self, memory, batch_size, step, episode_step, budget, fixed_indices=None):
        a = self.legacy
        if self.variant in LEGACY:
            batch = LegacyBatch(memory, fixed_indices)
            losses = a.update_parameters(batch, batch_size,
                                         self.counts['actor_updates'])
            ids = batch.sampled_indices
            if not np.isfinite(losses).all():
                raise FloatingPointError('Nonfinite historical SAC loss')
            result = {'critic_loss': losses[0]+losses[1], 'actor_loss': losses[2]}
        else:
            from replay import hre_window
            window, eta = hre_window(memory.size, step, episode_step, total_steps=budget,
                                     scale='ere' if self.priority_mode == 'standard' else 'legacy')
            beta = .4 + .6*min(1., step/budget)
            ids, actor_ids, weights = memory.draw(batch_size, self.variant, window, beta)
            tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
            s = tensor(memory.obs[ids]).reshape(batch_size, -1)
            ns = tensor(memory.next_obs[ids]).reshape(batch_size, -1)
            actions = tensor(memory.action[ids])
            rewards = tensor(memory.reward[ids]).unsqueeze(1)
            mask = 1. - tensor(memory.terminal[ids]).unsqueeze(1)
            with torch.no_grad():
                next_actions, next_logp, _ = a.policy.sample(ns)
                tq1, tq2 = a.critic_target(ns, next_actions)
                y = rewards + mask*a.gamma*(torch.minimum(tq1, tq2)-a.alpha*next_logp)
                if has_qclip(self.stabilizer):
                    y, a.qclip_active_fraction = clip_target(y, a.q_max)
            q1, q2 = a.critic(s, actions)
            td1, td2 = q1-y, q2-y
            critic_loss = (tensor(weights).unsqueeze(1)*(td1.square()+td2.square())).mean()
            if not torch.isfinite(critic_loss):
                raise FloatingPointError('Nonfinite weighted critic loss')
            a.critic_optim.zero_grad()
            critic_loss.backward()
            a.critic_optim.step()
            raw = torch.maximum(td1.abs(), td2.abs()).detach().flatten().cpu().numpy()
            if self.priority_mode == 'standard':
                transformed = memory.transformed_priorities(raw)  # standard priority: p = |delta| + 1e-6
            else:  # original expression kept bit-identical (float32 arithmetic)
                transformed = np.power(np.maximum(raw, 1.), .4) if self.priority_mode == 'floor' else np.power(raw+1e-6, .4)
            memory.update_priorities(ids, transformed)
            actor_states = tensor(memory.obs[actor_ids]).reshape(batch_size, -1)
            policy_actions, logp, _ = a.policy.sample(actor_states)
            pq1, pq2 = a.critic(actor_states, policy_actions)
            actor_loss = (a.alpha*logp-torch.minimum(pq1, pq2)).mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError('Nonfinite actor loss')
            a.policy_optim.zero_grad()
            actor_loss.backward()
            grad_norm = float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in a.policy.parameters() if p.grad is not None)))
            a.policy_optim.step()
            soft_update(a.critic_target, a.critic, a.tau)
            half = batch_size//2
            age = step - memory.steps[ids[half:]]
            saturation = float((policy_actions.detach().abs() > .99).float().mean())
            result = {'critic_loss': float(critic_loss.detach()), 'actor_loss': float(actor_loss.detach()),
                      'window': window, 'eta': eta, 'beta': beta,
                      'critic_sample_age_mean': float(age.mean()), 'critic_sample_age_median': float(np.median(age)),
                      'actor_grad_norm': grad_norm, 'action_saturation_fraction': saturation}
        if self.stabilizer != 'none':
            result['qclip_active_fraction'] = getattr(a, 'qclip_active_fraction', 0.)
            result['pbrs_shaping_mean'] = float(np.mean(memory.shaping[ids])) if has_pbrs(self.stabilizer) else 0.
        self.counts['actor_updates'] += 1
        self.counts['critic_optimizer_updates'] += 1
        self.counts['target_updates'] += 1
        self.counts['actor_samples'] += batch_size
        self.counts['critic_samples'] += batch_size
        return result

    def state_dict(self):
        a = self.legacy
        return {'variant': self.variant, 'seed': self.seed, 'priority_mode': self.priority_mode,
                'config': self.config, 'counts': self.counts.copy(), 'capacity_match': self.capacity_match,
                'policy_state_dict': a.policy.state_dict(), 'critic_state_dict': a.critic.state_dict(),
                'critic_target_state_dict': a.critic_target.state_dict(),
                'policy_optimizer_state_dict': a.policy_optim.state_dict(),
                'critic_optimizer_state_dict': a.critic_optim.state_dict(), 'alpha': a.alpha}

    def load_state_dict(self, d):
        if d['variant'] != self.variant or d['priority_mode'] != self.priority_mode:
            raise ValueError('Checkpoint algorithm mismatch')
        a = self.legacy
        for key, obj in [('policy_state_dict', a.policy), ('critic_state_dict', a.critic),
                         ('critic_target_state_dict', a.critic_target),
                         ('policy_optimizer_state_dict', a.policy_optim),
                         ('critic_optimizer_state_dict', a.critic_optim)]:
            if key in d:
                obj.load_state_dict(d[key])
        self.counts = d['counts'].copy()
        a.alpha = d['alpha']

    def diagnostics(self, memory, step):
        """Uniform probe, isolated all RNG, no training/replay mutations."""
        if memory.size == 0:
            return {}
        with isolated_rng(), torch.no_grad():
            r = np.random.default_rng(self.seed + int(step) + 80000000)
            ids = r.choice(memory.size, min(4096, memory.size), replace=False)
            errors = []
            for pos in range(0, len(ids), 256):
                ix = ids[pos:pos+256]
                tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
                s, ns = tensor(memory.obs[ix]).flatten(1), tensor(memory.next_obs[ix]).flatten(1)
                na, lp, _ = self.legacy.policy.sample(ns)
                q1, q2 = self.legacy.critic_target(ns, na)
                y = tensor(memory.reward[ix]).unsqueeze(1)+(1-tensor(memory.terminal[ix]).unsqueeze(1))*.99*(torch.minimum(q1,q2)-.025*lp)
                if has_qclip(self.stabilizer):
                    y, _ = clip_target(y, self.legacy.q_max)
                p1, p2 = self.legacy.critic(s, tensor(memory.action[ix]))
                errors.extend(torch.maximum((p1-y).abs(), (p2-y).abs()).flatten().cpu().tolist())
            e = np.asarray(errors)
            return {'uniform_td_probe_n': len(e), 'current_td_floor_fraction': float(np.mean(e<=1.)),
                    'current_td_quantiles': np.quantile(e, [0,.25,.5,.75,.9,.99,1]).tolist()}

    def parameter_counts(self):
        return {'actor': sum(p.numel() for p in self.policy.parameters()),
                'critic': sum(p.numel() for p in self.legacy.critic.parameters()),
                'target_critic': sum(p.numel() for p in self.legacy.critic_target.parameters())}
