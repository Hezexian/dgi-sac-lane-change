import torch
import torch.nn as nn
import torch.nn.functional as F


GRAPH_EPSILON = 1e-6


class WeightedGraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.self_linear = nn.Linear(in_channels, out_channels)
        self.neighbor_linear = nn.Linear(in_channels, out_channels)

    def forward(self, node_features, adjacency, active_mask=None):
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(GRAPH_EPSILON)
        aggregated = torch.matmul(adjacency, node_features)
        aggregated = aggregated / degree

        hidden = self.self_linear(node_features) + self.neighbor_linear(aggregated)
        if active_mask is not None:
            hidden = hidden * active_mask.unsqueeze(-1).to(hidden.dtype)
        return hidden


class GCNEncoder(nn.Module):
    def __init__(
        self,
        graph_hidden_size,
        graph_num_layers,
        edge_weight_mode,
        self_loop_weight,
        position_indices,
        node_feature_mode,
        use_presence_mask,
        distance_threshold,
        obs_feature_names,
        obs_presence_index,
        position_scale=None,
        ego_at_origin=False,
    ):
        super().__init__()
        # graph distances are computed in metres with the ego vehicle at the origin, so that
        # ego-neighbour inverse-distance edges carry physical meaning (see observation.py).
        self.position_scale = None if position_scale is None else tuple(float(v) for v in position_scale)
        self.ego_at_origin = bool(ego_at_origin)
        # Evaluation-only ablation hook: 'none', 'cut_ego_edges', 'ego_origin'. Never set during training.
        self.ablation = 'none'

        self.graph_hidden_size = graph_hidden_size
        self.graph_num_layers = max(1, int(graph_num_layers))
        self.edge_weight_mode = edge_weight_mode
        self.self_loop_weight = float(self_loop_weight)
        self.position_indices = tuple(position_indices)
        self.node_feature_mode = node_feature_mode
        self.use_presence_mask = bool(use_presence_mask)
        self.distance_threshold = distance_threshold
        self.obs_feature_names = list(obs_feature_names)
        self.obs_presence_index = obs_presence_index

        self.node_feature_indices = self._resolve_node_feature_indices()
        node_feature_dim = len(self.node_feature_indices)
        if node_feature_dim <= 0:
            raise ValueError("GCNEncoder requires at least one node feature")
        self.register_buffer(
            "node_feature_index_tensor",
            torch.tensor(self.node_feature_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "position_index_tensor",
            torch.tensor(self.position_indices, dtype=torch.long),
            persistent=False,
        )

        conv_layers = []
        in_channels = node_feature_dim
        for _ in range(self.graph_num_layers):
            conv_layers.append(WeightedGraphConvLayer(in_channels, graph_hidden_size))
            in_channels = graph_hidden_size
        self.convs = nn.ModuleList(conv_layers)

    def _resolve_node_feature_indices(self):
        if self.node_feature_mode != "all_except_presence":
            raise ValueError(
                f"Unsupported node_feature_mode '{self.node_feature_mode}'. "
                "First implementation only supports 'all_except_presence'."
            )

        if self.obs_presence_index is None:
            return list(range(len(self.obs_feature_names)))

        return [
            feature_index
            for feature_index in range(len(self.obs_feature_names))
            if feature_index != self.obs_presence_index
        ]

    def _build_active_mask(self, state_matrix):
        if not self.use_presence_mask or self.obs_presence_index is None:
            active_mask = torch.ones(
                state_matrix.size(0),
                state_matrix.size(1),
                dtype=torch.bool,
                device=state_matrix.device,
            )
        else:
            active_mask = state_matrix[:, :, self.obs_presence_index] > 0.5
            active_mask[:, 0] = True
        return active_mask

    def _build_adjacency(self, positions, active_mask):
        num_nodes = positions.size(1)
        distances = torch.cdist(positions, positions, p=2)
        active_pairs = active_mask.unsqueeze(1) & active_mask.unsqueeze(2)

        if self.edge_weight_mode == "uniform":
            adjacency = active_pairs.to(dtype=positions.dtype)
        elif self.edge_weight_mode == "inverse_distance":
            safe_distances = distances.clamp_min(GRAPH_EPSILON)
            adjacency = torch.where(
                active_pairs,
                1.0 / safe_distances,
                torch.zeros_like(distances),
            )
        else:
            raise ValueError(
                f"Unsupported edge_weight_mode '{self.edge_weight_mode}'. "
                "Supported modes: 'uniform', 'inverse_distance'."
            )

        diagonal_mask = torch.eye(num_nodes, dtype=torch.bool, device=positions.device).unsqueeze(0)
        adjacency = adjacency.masked_fill(diagonal_mask, 0.0)

        if self.distance_threshold is not None:
            threshold_mask = distances <= float(self.distance_threshold)
            adjacency = torch.where(threshold_mask, adjacency, torch.zeros_like(adjacency))

        self_loop_values = active_mask.to(dtype=positions.dtype) * self.self_loop_weight
        adjacency = adjacency.clone()
        diagonal_indices = torch.arange(num_nodes, device=positions.device)
        adjacency[:, diagonal_indices, diagonal_indices] = self_loop_values
        return adjacency

    def forward(self, state_matrix):
        if state_matrix.dim() != 3:
            raise ValueError(
                f"GCNEncoder expects [batch, vehicles, features], got shape {tuple(state_matrix.shape)}"
            )

        active_mask = self._build_active_mask(state_matrix)
        node_features = state_matrix.index_select(dim=2, index=self.node_feature_index_tensor)
        node_features = node_features * active_mask.unsqueeze(-1).to(node_features.dtype)
        positions = state_matrix.index_select(dim=2, index=self.position_index_tensor)
        if self.position_scale is not None:
            positions = positions * positions.new_tensor(self.position_scale)
        if self.ego_at_origin or self.ablation == 'ego_origin':
            positions = positions.clone()
            positions[:, 0, :] = 0.0
        adjacency = self._build_adjacency(positions, active_mask)
        if self.ablation == 'cut_ego_edges':
            adjacency = adjacency.clone()
            adjacency[:, 0, 1:] = 0.0
            adjacency[:, 1:, 0] = 0.0

        hidden = node_features
        for conv in self.convs:
            hidden = F.relu(conv(hidden, adjacency, active_mask))

        return hidden[:, 0, :]


__all__ = ["GCNEncoder", "WeightedGraphConvLayer"]
