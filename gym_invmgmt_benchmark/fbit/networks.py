from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.nn import GATv2Conv

from .config import FBiTConfig


def _group_sum(
    values: torch.Tensor, groups: torch.Tensor, num_groups: int
) -> torch.Tensor:
    result = torch.zeros(
        (num_groups, values.shape[-1]), dtype=values.dtype, device=values.device
    )
    return result.index_add(0, groups, values)


class DemandPredictor(nn.Module):
    """Causal Transformer forecasting one next value per retail demand link."""

    def __init__(self, config: FBiTConfig, num_retail: int):
        super().__init__()
        self.num_retail = int(num_retail)
        self.input_projection = nn.Sequential(
            nn.Linear(1, config.predictor_dim),
            nn.ReLU(),
            nn.Linear(config.predictor_dim, config.predictor_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.predictor_dim,
            nhead=config.predictor_heads,
            dim_feedforward=config.predictor_dim * 2,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.predictor_layers
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.predictor_dim),
            nn.Linear(config.predictor_dim, 1),
        )
        self.register_buffer(
            "position_encoding",
            self._position_encoding(config.num_periods, config.predictor_dim),
            persistent=False,
        )

    @staticmethod
    def _position_encoding(length: int, width: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32)
            * (-math.log(10000.0) / width)
        )
        encoding = torch.zeros(length, width)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding

    def forward(
        self, histories: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        # histories: [batch, periods, retail_links]
        batch, periods, retail = histories.shape
        if retail != self.num_retail:
            raise ValueError(f"Expected {self.num_retail} retail links, got {retail}")
        sequence = histories.permute(0, 2, 1).reshape(
            batch * retail, periods, 1
        )
        encoded = self.input_projection(sequence)
        encoded = encoded + self.position_encoding[:periods].unsqueeze(0)

        effective_lengths = lengths.clamp(min=1, max=periods)
        repeated_lengths = effective_lengths.repeat_interleave(retail)
        positions = torch.arange(periods, device=histories.device).unsqueeze(0)
        padding_mask = positions >= repeated_lengths.unsqueeze(1)
        causal_mask = torch.triu(
            torch.ones(periods, periods, dtype=torch.bool, device=histories.device),
            diagonal=1,
        )
        encoded = self.transformer(
            encoded, mask=causal_mask, src_key_padding_mask=padding_mask
        )
        row = torch.arange(batch * retail, device=histories.device)
        last = encoded[row, repeated_lengths - 1]
        prediction = F.softplus(self.output(last))
        return prediction.reshape(batch, retail)


class BidirectionalGraphEncoder(nn.Module):
    def __init__(self, config: FBiTConfig):
        super().__init__()
        hidden = config.hidden_dim
        heads = config.graph_heads
        output = config.graph_dim
        edge_dim = config.edge_feature_dim
        self.backward1 = GATv2Conv(
            config.node_feature_dim,
            hidden,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
        )
        self.backward2 = GATv2Conv(
            hidden * heads,
            output,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
        )
        self.forward1 = GATv2Conv(
            output,
            hidden,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
        )
        self.forward2 = GATv2Conv(
            hidden * heads,
            output,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, graph, predicted_node_demand: torch.Tensor) -> torch.Tensor:
        x = torch.cat((graph.x, predicted_node_demand), dim=1)
        backward = self.activation(
            self.backward1(x, graph.reverse_edge_index, graph.edge_attr)
        )
        backward = self.activation(
            self.backward2(
                backward, graph.reverse_edge_index, graph.edge_attr
            )
        )
        forward = self.activation(
            self.forward1(backward, graph.edge_index, graph.edge_attr)
        )
        forward = self.activation(
            self.forward2(forward, graph.edge_index, graph.edge_attr)
        )
        return torch.cat((forward, backward), dim=1)


class PolicyNetwork(nn.Module):
    def __init__(self, config: FBiTConfig):
        super().__init__()
        # Encoder receives the predicted demand as one extra node feature.
        encoder_config = FBiTConfig.from_dict(config.to_dict())
        encoder_config.node_feature_dim += 1
        self.encoder = BidirectionalGraphEncoder(encoder_config)
        input_dim = config.graph_dim * 4 + 1
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, config.actor_hidden),
            nn.ReLU(),
            nn.Linear(config.actor_hidden, config.actor_hidden // 2),
            nn.ReLU(),
        )
        self.mu = nn.Linear(config.actor_hidden // 2, 1)
        self.log_std = nn.Linear(config.actor_hidden // 2, 1)

    def forward(
        self, graph, predicted_node_demand: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nodes = self.encoder(graph, predicted_node_demand)
        src, dst = graph.action_edge_index
        features = torch.cat(
            (nodes[src], nodes[dst], graph.pipeline.reshape(-1, 1)), dim=1
        )
        hidden = self.trunk(features)
        mu = self.mu(hidden)
        log_std = self.log_std(hidden).clamp(-10.0, 2.0)
        distribution = Normal(mu, log_std.exp())
        latent = mu if deterministic else distribution.rsample()
        action = torch.tanh(latent)
        per_action_log_prob = distribution.log_prob(latent) - torch.log(
            1.0 - action.pow(2) + 1e-7
        )
        action_groups = graph.batch[src]
        log_prob = _group_sum(
            per_action_log_prob, action_groups, graph.num_graphs
        )
        return action.reshape(-1), log_prob


class QNetwork(nn.Module):
    def __init__(self, config: FBiTConfig, num_nodes: int, num_actions: int):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_actions = int(num_actions)
        encoder_config = FBiTConfig.from_dict(config.to_dict())
        encoder_config.node_feature_dim += 1
        self.encoder = BidirectionalGraphEncoder(encoder_config)
        node_dim = config.graph_dim * 2
        self.fc_s = nn.Linear(
            self.num_nodes * node_dim, config.critic_hidden
        )
        self.fc_a = nn.Linear(self.num_actions, config.critic_hidden)
        self.fc_cat = nn.Linear(
            config.critic_hidden * 2, config.critic_hidden
        )
        self.fc_out = nn.Linear(config.critic_hidden, 1)

    def forward(
        self, graph, action: torch.Tensor, predicted_node_demand: torch.Tensor
    ) -> torch.Tensor:
        nodes = self.encoder(graph, predicted_node_demand)
        batch_size = int(graph.num_graphs)
        expected_nodes = batch_size * self.num_nodes
        expected_actions = batch_size * self.num_actions
        if nodes.shape[0] != expected_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes per graph "
                f"({expected_nodes} total), got {nodes.shape[0]}"
            )
        if action.numel() != expected_actions:
            raise ValueError(
                f"Expected {expected_actions} action values, got {action.numel()}"
            )

        state_graph = nodes.reshape(batch_size, -1)
        action_graph = action.reshape(batch_size, self.num_actions)
        state_hidden = self.fc_s(state_graph)
        action_hidden = self.fc_a(action_graph)
        joint = torch.cat((state_hidden, action_hidden), dim=1)
        return self.fc_out(F.relu(self.fc_cat(joint)))


class FBiTSystem(nn.Module):
    def __init__(
        self,
        config: FBiTConfig,
        num_retail: int,
        retailer_node_indices: Sequence[int],
        num_nodes: int,
        num_actions: int,
    ):
        super().__init__()
        self.config = config
        self.num_retail = int(num_retail)
        self.retailer_node_indices = tuple(int(x) for x in retailer_node_indices)
        self.predictor = DemandPredictor(config, num_retail)
        self.actor = PolicyNetwork(config)
        self.q1 = QNetwork(config, num_nodes, num_actions)
        self.q2 = QNetwork(config, num_nodes, num_actions)
        self.q1_target = QNetwork(config, num_nodes, num_actions)
        self.q2_target = QNetwork(config, num_nodes, num_actions)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(config.initial_alpha), dtype=torch.float32)
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def predict_retail_demand(self, graph) -> torch.Tensor:
        histories = graph.demand_history
        if histories.ndim == 2:
            histories = histories.unsqueeze(0)
        lengths = graph.history_length.reshape(-1)
        return self.predictor(histories, lengths)

    def demand_to_nodes(self, graph, retail_demand: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(
            (graph.num_nodes, 1), dtype=retail_demand.dtype, device=retail_demand.device
        )
        for graph_idx in range(graph.num_graphs):
            offset = int(graph.ptr[graph_idx])
            for retail_idx, node_idx in enumerate(self.retailer_node_indices):
                output[offset + node_idx, 0] += retail_demand[graph_idx, retail_idx]
        return output

    def soft_update_targets(self) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for source, target in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for source_param, target_param in zip(
                    source.parameters(), target.parameters()
                ):
                    target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)
