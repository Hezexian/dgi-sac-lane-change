"""SAC with an actor combining flattened and vehicle-graph branches.

Graph edge weights follow an inverse-distance rule. The two branch
representations are concatenated and fused before the Gaussian policy head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..networks.actors import GaussianActorBase
from ..networks.critics import weights_init_
from .sac import SACAgent

GRAPH_EPSILON = 1e-6


class WeightedGraphConvLayer(nn.Module):
    """h_i' = W_self h_i + W_nbr * sum_j w_ij h_j / sum_j w_ij   (Eq. 3; the sum includes the self-loop)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.self_linear = nn.Linear(in_channels, out_channels)
        self.neighbor_linear = nn.Linear(in_channels, out_channels)

    def forward(self, node_features, adjacency, active_mask=None):
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(GRAPH_EPSILON)
        aggregated = torch.matmul(adjacency, node_features) / degree
        hidden = self.self_linear(node_features) + self.neighbor_linear(aggregated)
        if active_mask is not None:
            hidden = hidden * active_mask.unsqueeze(-1).to(hidden.dtype)
        return hidden


class GraphEncoder(nn.Module):
    """Graph branch. Input [batch, vehicles, features]; output: ego-node representation [batch, hidden]."""

    def __init__(self, hidden_size=256, num_layers=3,
                 feature_names=("presence", "x", "y", "vx", "vy", "heading"),
                 position_scale=(100.0, 12.0), ego_at_origin=True, self_loop_weight=1.0):
        super().__init__()
        self.feature_names = list(feature_names)
        self.presence_index = self.feature_names.index("presence")
        self.position_indices = (self.feature_names.index("x"), self.feature_names.index("y"))
        self.position_scale = tuple(float(v) for v in position_scale)  # back to metres
        self.ego_at_origin = bool(ego_at_origin)
        self.self_loop_weight = float(self_loop_weight)
        # Evaluation-only intervention of Section 4.3: 'cut_ego_edges' removes the ego-neighbour edges.
        self.ablation = "none"
        node_features = [i for i in range(len(self.feature_names)) if i != self.presence_index]
        self.register_buffer("node_feature_index_tensor", torch.tensor(node_features, dtype=torch.long),
                             persistent=False)
        self.register_buffer("position_index_tensor", torch.tensor(self.position_indices, dtype=torch.long),
                             persistent=False)
        layers, in_channels = [], len(node_features)
        for _ in range(max(1, int(num_layers))):
            layers.append(WeightedGraphConvLayer(in_channels, hidden_size))
            in_channels = hidden_size
        self.convs = nn.ModuleList(layers)

    def adjacency(self, positions, active_mask):
        """Fixed inverse-distance edge weights between present vehicles; self-loops of weight 1 (Eq. 2)."""
        num_nodes = positions.size(1)
        distances = torch.cdist(positions, positions, p=2)
        active_pairs = active_mask.unsqueeze(1) & active_mask.unsqueeze(2)
        adjacency = torch.where(active_pairs, 1.0 / distances.clamp_min(GRAPH_EPSILON), torch.zeros_like(distances))
        diagonal = torch.eye(num_nodes, dtype=torch.bool, device=positions.device).unsqueeze(0)
        adjacency = adjacency.masked_fill(diagonal, 0.0).clone()
        index = torch.arange(num_nodes, device=positions.device)
        adjacency[:, index, index] = active_mask.to(positions.dtype) * self.self_loop_weight
        return adjacency

    def forward(self, state_matrix):
        active_mask = state_matrix[:, :, self.presence_index] > 0.5
        active_mask[:, 0] = True  # the ego vehicle is always present
        node_features = state_matrix.index_select(2, self.node_feature_index_tensor)
        node_features = node_features * active_mask.unsqueeze(-1).to(node_features.dtype)
        positions = state_matrix.index_select(2, self.position_index_tensor)
        positions = positions * positions.new_tensor(self.position_scale)
        if self.ego_at_origin:
            positions = positions.clone()
            positions[:, 0, :] = 0.0
        adjacency = self.adjacency(positions, active_mask)
        if self.ablation == "cut_ego_edges":
            adjacency = adjacency.clone()
            adjacency[:, 0, 1:] = 0.0
            adjacency[:, 1:, 0] = 0.0
        hidden = node_features
        for conv in self.convs:
            hidden = F.relu(conv(hidden, adjacency, active_mask))
        return hidden[:, 0, :]


class GraphActor(GaussianActorBase):
    """DGI-SAC actor: [flattened branch, graph branch] -> fusion layer -> Gaussian heads (Fig. 1)."""

    def __init__(self, num_inputs, num_actions, hidden_size, action_space=None, *, vehicles=8,
                 feature_names=("presence", "x", "y", "vx", "vy", "heading"), graph=None):
        super().__init__()
        graph = dict(graph or {})
        self.vehicles, self.features = int(vehicles), len(feature_names)
        if self.vehicles * self.features != num_inputs:
            raise ValueError("Observation size does not match vehicles x features")
        graph_hidden = int(graph.get("hidden_size", hidden_size))
        self.linear1 = nn.Linear(num_inputs, hidden_size)          # flattened branch
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.graph_encoder = GraphEncoder(                          # graph branch
            hidden_size=graph_hidden, num_layers=graph.get("num_layers", 3), feature_names=feature_names,
            position_scale=graph.get("position_scale", (100.0, 12.0)),
            ego_at_origin=graph.get("ego_at_origin", True), self_loop_weight=graph.get("self_loop_weight", 1.0))
        self.fusion = nn.Linear(hidden_size + graph_hidden, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.apply(weights_init_)
        self._set_action_range(action_space)
        # Evaluation-only intervention: 'zero_mlp_neighbors' hides neighbours from the flattened branch.
        self.ablation = "none"

    def forward(self, state):
        flat_input = state
        if self.ablation == "zero_mlp_neighbors":
            flat_input = state.clone()
            flat_input[:, self.features:] = 0.0
        x_flat = F.relu(self.linear2(F.relu(self.linear1(flat_input))))
        x_graph = self.graph_encoder(state.reshape(state.size(0), self.vehicles, self.features))
        x = F.relu(self.fusion(torch.cat([x_flat, x_graph], dim=1)))
        return self._gaussian(x)


def make_graph_actor(config, num_inputs, num_actions, action_space):
    observation = config["env"]["observation"]
    return GraphActor(num_inputs, num_actions, config["train"]["hidden_size"], action_space,
                      vehicles=observation["vehicles_count"], feature_names=observation["features"],
                      graph=config.get("graph"))


class DGISACAgent(SACAgent):
    """SAC with the graph actor; everything else is inherited from SACAgent."""

    def build_actor(self, num_inputs, num_actions, action_space):
        return make_graph_actor(self.config, num_inputs, num_actions, action_space)
