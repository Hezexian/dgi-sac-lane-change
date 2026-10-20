import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .encoders import GCNEncoder


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


def weights_init_(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, gain=1)
        torch.nn.init.constant_(module.bias, 0)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_actions,
        hidden_dim,
        action_space=None,
        algorithm_params=None,
        obs_vehicles_count=None,
        obs_feature_dim=None,
        obs_feature_names=None,
        obs_presence_index=None,
    ):
        super(GaussianPolicy, self).__init__()

        self.algorithm_params = algorithm_params or {}
        self.obs_vehicles_count = obs_vehicles_count
        self.obs_feature_dim = obs_feature_dim
        self.obs_feature_names = list(obs_feature_names or [])
        self.obs_presence_index = obs_presence_index
        self.obs_flat_dim = (
            self.obs_vehicles_count * self.obs_feature_dim
            if self.obs_vehicles_count is not None and self.obs_feature_dim is not None
            else None
        )

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.graph_encoder = GCNEncoder(
            graph_hidden_size=self.algorithm_params.get("graph_hidden_size", hidden_dim),
            graph_num_layers=self.algorithm_params.get("graph_num_layers", 1),
            edge_weight_mode=self.algorithm_params.get("edge_weight_mode", "inverse_distance"),
            self_loop_weight=self.algorithm_params.get("self_loop_weight", 1.0),
            position_indices=self.algorithm_params.get("position_indices", [1, 2]),
            node_feature_mode=self.algorithm_params.get("node_feature_mode", "all_except_presence"),
            use_presence_mask=self.algorithm_params.get("use_presence_mask", True),
            distance_threshold=self.algorithm_params.get("distance_threshold"),
            obs_feature_names=self.obs_feature_names,
            obs_presence_index=self.obs_presence_index,
            position_scale=self.algorithm_params.get("position_scale"),
            ego_at_origin=self.algorithm_params.get("ego_at_origin", False),
        )
        # Evaluation-only ablation hook ('none' or 'zero_mlp_neighbors'); never set during training.
        self.ablation = 'none'
        graph_hidden_size = self.algorithm_params.get("graph_hidden_size", hidden_dim)
        fusion_mode = self.algorithm_params.get("fusion_mode", "concat")
        if fusion_mode != "concat":
            raise ValueError(
                f"Unsupported fusion_mode '{fusion_mode}'. First implementation only supports 'concat'."
            )
        self.fusion_mode = fusion_mode
        self.fusion = nn.Linear(hidden_dim + graph_hidden_size, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.0)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.0)

    def _reshape_flat_state(self, state):
        if self.obs_flat_dim is None:
            raise ValueError("Observation metadata is incomplete for sac_gcn policy")
        if state.size(1) != self.obs_flat_dim:
            raise ValueError(
                f"Expected flattened state dim {self.obs_flat_dim}, got {state.size(1)}. "
                "Check env.observation config and config_loader metadata."
            )
        return state.reshape(state.size(0), self.obs_vehicles_count, self.obs_feature_dim)

    def forward(self, state):
        state_matrix = self._reshape_flat_state(state)

        mlp_input = state
        if self.ablation == 'zero_mlp_neighbors':
            mlp_input = state.clone()
            mlp_input[:, self.obs_feature_dim:] = 0.0
        x_mlp = F.relu(self.linear1(mlp_input))
        x_mlp = F.relu(self.linear2(x_mlp))

        x_graph = self.graph_encoder(state_matrix)

        x = torch.cat([x_mlp, x_graph], dim=1)
        x = F.relu(self.fusion(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # tanh change of variables (Haarnoja et al. 2018, App. C) in the stable form log(1-tanh(x)^2) = 2(log2 - x - softplus(-2x)) used by OpenAI Spinning Up SAC
        log_prob -= torch.log(self.action_scale) + 2.0 * (math.log(2.0) - x_t - F.softplus(-2.0 * x_t))
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.0)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.0)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0.0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.0), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
