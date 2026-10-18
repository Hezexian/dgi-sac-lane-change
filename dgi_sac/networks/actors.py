import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .critics import final_layer_init, weights_init_

LOG_STD_MIN, LOG_STD_MAX = -20, 2


class GaussianActorBase(nn.Module):

    def _set_action_range(self, action_space):
        if action_space is None:
            self.action_scale, self.action_bias = torch.tensor(1.0), torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.0)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.0)

    def _gaussian(self, features):
        mean = self.mean_linear(features)
        log_std = self.log_std_linear(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """Return (sampled action, log-probability, deterministic action)."""
        mean, log_std = self.forward(state)
        normal = Normal(mean, log_std.exp())
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.action_scale + self.action_bias
        # log(1 - tanh(x)^2) = 2 (log 2 - x - softplus(-2x))
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale) + 2.0 * (math.log(2.0) - x_t - F.softplus(-2.0 * x_t))
        log_prob = log_prob.sum(1, keepdim=True)
        deterministic = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class GaussianMLPActor(GaussianActorBase):
    """Two fully connected layers on the flattened observation (SAC, capacity-matched MLP)."""

    def __init__(self, num_inputs, num_actions, hidden_size, action_space=None):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.apply(weights_init_)
        self._set_action_range(action_space)

    def forward(self, state):
        x = F.relu(self.linear2(F.relu(self.linear1(state))))
        return self._gaussian(x)


class DeterministicActor(nn.Module):

    def __init__(self, num_inputs=48, num_actions=2, hidden_size=256):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.apply(weights_init_)
        final_layer_init(self.linear3)

    def forward(self, state):
        return torch.tanh(self.linear3(F.relu(self.linear2(F.relu(self.linear1(state))))))
