"""TD3 (Fujimoto et al., 2018): twin critics, target-policy smoothing and delayed actor updates."""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from ..config import obs_dim
from ..networks import DeterministicActor, DeterministicCritic
from ..utils import seed_everything, soft_update


class TD3Agent:
    def __init__(self, config, seed, action_space, device="cpu", threads=1):
        seed_everything(seed, threads)
        self.config, self.seed = config, seed
        self.device = torch.device(device)
        train = config["train"]
        self.gamma, self.tau = train["gamma"], train["tau"]
        self.exploration_noise = train["exploration_noise"]
        self.target_noise, self.noise_clip = train["target_noise"], train["noise_clip"]
        self.policy_delay = train["policy_delay"]
        n_obs, n_act, hidden = obs_dim(config), action_space.shape[0], train["hidden_size"]

        self.policy = DeterministicActor(n_obs, n_act, hidden).to(self.device)
        self.critic = DeterministicCritic(n_obs, n_act, hidden, twin=True).to(self.device)
        self.policy_target = copy.deepcopy(self.policy)
        self.critic_target = copy.deepcopy(self.critic)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=train["lr"])
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=train["lr"])
        self.exploration_rng = np.random.default_rng(seed)  # exploration noise
        self.updates = 0  # critic updates

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.as_tensor(np.asarray(state).reshape(1, -1), dtype=torch.float32, device=self.device)
            action = self.policy(state).cpu().numpy()[0]
        if not evaluate:
            noise = self.exploration_rng.normal(0.0, self.exploration_noise, action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    def update(self, memory, batch_size, indices=None):
        s, a, r, s2, not_done = (torch.as_tensor(x, dtype=torch.float32, device=self.device)
                                 for x in memory.sample(batch_size, indices))

        # Critic: y = r + gamma (1 - d) min_k Q_k'(s', clip(mu'(s') + clip(eps, -c, c), -1, 1))
        with torch.no_grad():
            a2 = self.policy_target(s2)
            noise = (torch.randn_like(a2) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = (a2 + noise).clamp(-1.0, 1.0)
            q_next = torch.stack(self.critic_target(s2, a2)).amin(dim=0)
            y = r.reshape(-1, 1) + not_done.reshape(-1, 1) * self.gamma * q_next
        critic_loss = sum(F.mse_loss(q, y) for q in self.critic(s, a))
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self.updates += 1
        losses = {"critic_loss": float(critic_loss.detach())}

        # Delayed actor and target updates: maximise Q_1(s, mu(s)) every policy_delay critic steps.
        if self.updates % self.policy_delay == 0:
            actor_loss = -self.critic(s, self.policy(s))[0].mean()
            self.policy_optim.zero_grad()
            actor_loss.backward()
            self.policy_optim.step()
            soft_update(self.policy_target, self.policy, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
            losses["actor_loss"] = float(actor_loss.detach())
        if not np.isfinite(list(losses.values())).all():
            raise FloatingPointError("Non-finite TD3 loss")
        return losses

    def parameter_counts(self):
        return {"actor": sum(p.numel() for p in self.policy.parameters()),
                "critic": sum(p.numel() for p in self.critic.parameters())}

    def state_dict(self):
        return {
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "policy_target_state_dict": self.policy_target.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            "critic_optimizer_state_dict": self.critic_optim.state_dict(),
            "exploration_rng": copy.deepcopy(self.exploration_rng.bit_generator.state),
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        for key, module in [("policy_state_dict", self.policy), ("critic_state_dict", self.critic),
                            ("policy_target_state_dict", self.policy_target),
                            ("critic_target_state_dict", self.critic_target),
                            ("policy_optimizer_state_dict", self.policy_optim),
                            ("critic_optimizer_state_dict", self.critic_optim)]:
            if key in state:
                module.load_state_dict(state[key])
        if "exploration_rng" in state:
            self.exploration_rng.bit_generator.state = copy.deepcopy(state["exploration_rng"])
        self.updates = state.get("updates", self.updates)
