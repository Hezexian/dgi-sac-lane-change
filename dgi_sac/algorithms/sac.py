"""SAC with a fixed entropy coefficient and optional TD-target clipping."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ..config import obs_dim
from ..networks import GaussianMLPActor, QNetwork
from ..utils import hard_update, seed_everything, soft_update


def td_target_bound(config):
    """Upper bound on the expected entropy-regularised return: (r_max + alpha log 4) / (1 - gamma).

    r_max is the largest per-step reward: each highway-env reward term lies in [0, 1], so r_max is the
    sum of the positive weights (0.5 here). log 4 bounds the entropy term of an action in [-1, 1]^2.
    """
    env, train = config["env"], config["train"]
    weights = ("collision_reward", "right_lane_reward", "high_speed_reward", "lane_change_reward")
    r_max = sum(max(0.0, float(env.get(k, 0.0))) for k in weights)
    return (r_max + float(train["alpha"]) * math.log(4.0)) / (1.0 - float(train["gamma"]))


class SACAgent:
    def __init__(self, config, seed, action_space, device="cpu", threads=1):
        seed_everything(seed, threads)
        self.config, self.seed = config, seed
        self.device = torch.device(device)
        train = config["train"]
        self.gamma, self.tau, self.alpha = train["gamma"], train["tau"], train["alpha"]
        clip = train.get("td_target_clip")
        self.td_target_clip = td_target_bound(config) if clip == "auto" else clip
        n_obs, n_act, hidden = obs_dim(config), action_space.shape[0], train["hidden_size"]

        # Construction order is fixed so that a seed reproduces the published initial weights.
        self.critic = QNetwork(n_obs, n_act, hidden).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=train["lr"])
        self.critic_target = QNetwork(n_obs, n_act, hidden).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.policy = self.build_actor(n_obs, n_act, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=train["lr"])
        self.updates = 0

    def build_actor(self, num_inputs, num_actions, action_space):
        return GaussianMLPActor(num_inputs, num_actions, self.config["train"]["hidden_size"], action_space)

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(np.asarray(state).reshape(-1)).to(self.device).unsqueeze(0)
            sampled, _, deterministic = self.policy.sample(state)
            action = deterministic if evaluate else sampled
            return action.detach().cpu().numpy()[0]

    def update(self, memory, batch_size, indices=None):
        s, a, r, s2, not_done = memory.sample(batch_size, indices)
        s = torch.FloatTensor(s).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device).unsqueeze(1)
        not_done = torch.FloatTensor(not_done).to(self.device).unsqueeze(1)

        # Critic: y = r + gamma (1 - d) [min_k Q_k'(s', a') - alpha log pi(a'|s')],  a' ~ pi(.|s')
        with torch.no_grad():
            a2, log_pi2, _ = self.policy.sample(s2)
            q1_next, q2_next = self.critic_target(s2, a2)
            y = r + not_done * self.gamma * (torch.min(q1_next, q2_next) - self.alpha * log_pi2)
            clipped_fraction = 0.0
            if self.td_target_clip is not None:
                clipped_fraction = float((y > self.td_target_clip).float().mean())
                y = y.clamp(max=self.td_target_clip)
        q1, q2 = self.critic(s, a)
        q1_loss, q2_loss = F.mse_loss(q1, y), F.mse_loss(q2, y)
        self.critic_optim.zero_grad()
        (q1_loss + q2_loss).backward()
        self.critic_optim.step()

        # Actor: minimise alpha log pi(a|s) - min_k Q_k(s, a),  a ~ pi(.|s) (reparameterised)
        pi, log_pi, _ = self.policy.sample(s)
        q1_pi, q2_pi = self.critic(s, pi)
        actor_loss = (self.alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()
        self.policy_optim.zero_grad()
        actor_loss.backward()
        self.policy_optim.step()

        soft_update(self.critic_target, self.critic, self.tau)
        self.updates += 1
        losses = {"critic_loss": q1_loss.item() + q2_loss.item(), "actor_loss": actor_loss.item()}
        if self.td_target_clip is not None:
            losses["clipped_target_fraction"] = clipped_fraction
        if not np.isfinite(list(losses.values())).all():
            raise FloatingPointError("Non-finite SAC loss")
        return losses

    def parameter_counts(self):
        return {"actor": sum(p.numel() for p in self.policy.parameters()),
                "critic": sum(p.numel() for p in self.critic.parameters())}

    def state_dict(self):
        return {
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            "critic_optimizer_state_dict": self.critic_optim.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        for key, module in [("policy_state_dict", self.policy), ("critic_state_dict", self.critic),
                            ("critic_target_state_dict", self.critic_target),
                            ("policy_optimizer_state_dict", self.policy_optim),
                            ("critic_optimizer_state_dict", self.critic_optim)]:
            if key in state:
                module.load_state_dict(state[key])
        self.updates = state.get("updates", self.updates)
