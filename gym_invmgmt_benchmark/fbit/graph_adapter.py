from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch_geometric.data import Data


PIPELINE_ARRIVAL_HORIZON = 12


NODE_FEATURE_NAMES = (
    "is_market",
    "is_retailer",
    "is_distributor",
    "is_factory",
    "is_raw_material",
    "inventory",
    "last_demand",
    "backlog",
    *(f"incoming_t{offset}" for offset in range(PIPELINE_ARRIVAL_HORIZON)),
    "outgoing_pipeline",
    "capacity",
    "holding_cost",
    "time",
    "sentiment",
)

EDGE_FEATURE_NAMES = (
    "is_reorder",
    "is_retail",
    "lead_time",
    "price",
    "pipeline_holding_cost",
)


@dataclass(frozen=True)
class ObservationParts:
    demand: np.ndarray
    backlog: np.ndarray
    inventory: np.ndarray
    pipeline_buckets: np.ndarray
    time: float
    sentiment: float


class CoreEnvGraphAdapter:
    """Convert public CoreEnv observations and static topology into PyG data.

    No demand-engine configuration, future demand matrix, or future order data
    is read here. Dynamic features are derived only from the observation passed
    by the caller.
    """

    def __init__(self, env, device: torch.device | str = "cpu"):
        self.env = env.unwrapped
        self.network = self.env.network
        self.device = torch.device(device)
        self.num_periods = int(self.env.num_periods)
        self.nodes = list(self.network.all_nodes)
        self.node_to_local = {node: idx for idx, node in enumerate(self.nodes)}
        self.reorder_links = list(self.network.reorder_links)
        self.network_links = list(self.network.network_links)
        self.retail_links = list(self.network.retail_links)
        self.num_nodes = len(self.nodes)
        self.num_actions = len(self.reorder_links)
        self.num_retail = len(self.retail_links)
        self.node_feature_dim = len(NODE_FEATURE_NAMES)
        self.pipeline_length = int(self.network.pipeline_length)
        self.pipeline_arrival_horizon = PIPELINE_ARRIVAL_HORIZON
        if int(self.network.lt_max) > self.pipeline_arrival_horizon:
            raise ValueError(
                f"Network lead time {self.network.lt_max} exceeds the "
                f"{self.pipeline_arrival_horizon}-period pipeline arrival horizon"
            )

        expected = (
            2 * self.num_retail
            + len(self.network.main_nodes)
            + self.pipeline_length
            + 2
        )
        if expected != int(self.env.observation_space.shape[0]):
            raise ValueError(
                f"Unexpected CoreEnv observation layout: computed {expected}, "
                f"space has {self.env.observation_space.shape[0]}"
            )

        self._edge_index = self._make_edge_index(self.network_links)
        self._reverse_edge_index = self._edge_index.flip(0)
        self._action_edge_index = self._make_edge_index(self.reorder_links)
        self._edge_attr = self._make_edge_features()

    def _make_edge_index(self, edges: Iterable[tuple]) -> torch.Tensor:
        values = [
            [self.node_to_local[src], self.node_to_local[dst]]
            for src, dst in edges
        ]
        return torch.as_tensor(values, dtype=torch.long).t().contiguous()

    def split_observation(self, observation: np.ndarray) -> ObservationParts:
        obs = np.asarray(observation, dtype=np.float64).reshape(-1)
        cursor = 0
        demand = obs[cursor : cursor + self.num_retail]
        cursor += self.num_retail
        backlog = obs[cursor : cursor + self.num_retail]
        cursor += self.num_retail
        inventory = obs[cursor : cursor + len(self.network.main_nodes)]
        cursor += len(self.network.main_nodes)
        pipeline = obs[cursor : cursor + self.pipeline_length]
        cursor += self.pipeline_length
        if cursor + 2 != obs.size:
            raise ValueError(
                f"Observation has {obs.size} values; parsed {cursor + 2}"
            )
        return ObservationParts(
            demand=demand.copy(),
            backlog=backlog.copy(),
            inventory=inventory.copy(),
            pipeline_buckets=pipeline.copy(),
            time=float(obs[cursor]),
            sentiment=float(obs[cursor + 1]),
        )

    def action_aligned_arrivals(self, buckets: np.ndarray) -> np.ndarray:
        buckets = np.asarray(buckets, dtype=np.float64).reshape(-1)
        if buckets.size != self.pipeline_length:
            raise ValueError(
                f"Expected {self.pipeline_length} pipeline buckets, got {buckets.size}"
            )
        result = np.zeros(
            (self.num_actions, PIPELINE_ARRIVAL_HORIZON), dtype=np.float64
        )
        cursor = 0
        for idx, edge in enumerate(self.reorder_links):
            lead_time = int(self.network.lead_times[edge])
            if lead_time:
                result[idx, :lead_time] = buckets[cursor : cursor + lead_time]
                cursor += lead_time
        if cursor != buckets.size:
            raise ValueError(
                f"Pipeline parser consumed {cursor} of {buckets.size} buckets"
            )
        return result

    def action_aligned_pipeline(self, buckets: np.ndarray) -> np.ndarray:
        return self.action_aligned_arrivals(buckets).sum(axis=1)

    def _make_edge_features(self) -> torch.Tensor:
        max_lead = max(1.0, float(self.network.lt_max))
        prices = [float(self.network.graph.edges[edge].get("p", 0.0)) for edge in self.network_links]
        max_price = max(1.0, max(prices, default=1.0))
        holding = [float(self.network.graph.edges[edge].get("g", 0.0)) for edge in self.network_links]
        max_holding = max(1e-6, max(holding, default=1e-6))
        rows = []
        reorder = set(self.reorder_links)
        for edge in self.network_links:
            attrs = self.network.graph.edges[edge]
            is_reorder = edge in reorder
            rows.append(
                [
                    float(is_reorder),
                    float(not is_reorder),
                    float(attrs.get("L", 0.0)) / max_lead,
                    float(attrs.get("p", 0.0)) / max_price,
                    float(attrs.get("g", 0.0)) / max_holding,
                ]
            )
        return torch.tensor(rows, dtype=torch.float32)

    def _node_type(self, node) -> List[float]:
        return [
            float(node in set(self.network.market)),
            float(node in set(self.network.retail)),
            float(node in set(self.network.distrib)),
            float(node in set(self.network.factory)),
            float(node in set(self.network.rawmat)),
        ]

    def _node_features(
        self,
        parts: ObservationParts,
        pipeline: np.ndarray,
        arrivals: np.ndarray,
    ) -> torch.Tensor:
        inventory_by_node = {
            node: float(value)
            for node, value in zip(self.network.main_nodes, parts.inventory)
        }
        demand_by_node = {}
        backlog_by_node = {}
        for (retailer, _market), demand, backlog in zip(
            self.retail_links, parts.demand, parts.backlog
        ):
            demand_by_node[retailer] = demand_by_node.get(retailer, 0.0) + float(demand)
            backlog_by_node[retailer] = backlog_by_node.get(retailer, 0.0) + float(backlog)

        incoming = {
            node: np.zeros(PIPELINE_ARRIVAL_HORIZON, dtype=np.float64)
            for node in self.nodes
        }
        outgoing = {node: 0.0 for node in self.nodes}
        for (supplier, purchaser), value, schedule in zip(
            self.reorder_links, pipeline, arrivals
        ):
            incoming[purchaser] += schedule
            outgoing[supplier] += float(value)

        inv_scale = max(1.0, float(self.network.init_inv_max))
        pipeline_scale = max(
            1.0, float(np.max(np.asarray(self.env.action_space.high)))
        )
        capacity_scale = max(1.0, float(self.network.capacity_max))
        holding_values = [
            float(self.network.graph.nodes[node].get("h", 0.0))
            for node in self.nodes
        ]
        holding_scale = max(1e-6, max(holding_values, default=1e-6))
        rows = []
        for node in self.nodes:
            attrs = self.network.graph.nodes[node]
            rows.append(
                self._node_type(node)
                + [
                    inventory_by_node.get(node, 0.0) / inv_scale,
                    demand_by_node.get(node, 0.0) / 20.0,
                    backlog_by_node.get(node, 0.0) / 20.0,
                ]
                + (incoming[node] / pipeline_scale).tolist()
                + [
                    outgoing[node] / pipeline_scale,
                    float(attrs.get("C", 0.0)) / capacity_scale,
                    float(attrs.get("h", 0.0)) / holding_scale,
                    parts.time / max(1.0, float(self.num_periods)),
                    parts.sentiment / 2.0,
                ]
            )
        return torch.tensor(rows, dtype=torch.float32)

    def graph(
        self,
        observation: np.ndarray,
        demand_history: Sequence[np.ndarray],
    ) -> Data:
        parts = self.split_observation(observation)
        arrivals = self.action_aligned_arrivals(parts.pipeline_buckets)
        pipeline = arrivals.sum(axis=1)
        history = np.zeros(
            (self.num_periods, self.num_retail), dtype=np.float32
        )
        history_length = min(len(demand_history), self.num_periods)
        if history_length:
            history[:history_length] = np.asarray(
                demand_history[-history_length:], dtype=np.float32
            )
        return Data(
            x=self._node_features(parts, pipeline, arrivals),
            edge_index=self._edge_index.clone(),
            reverse_edge_index=self._reverse_edge_index.clone(),
            edge_attr=self._edge_attr.clone(),
            action_edge_index=self._action_edge_index.clone(),
            pipeline=torch.tensor(pipeline[:, None], dtype=torch.float32),
            demand_history=torch.tensor(history[None, ...], dtype=torch.float32),
            history_length=torch.tensor([history_length], dtype=torch.long),
        ).to(self.device)

    def observed_demand(self, observation: np.ndarray) -> np.ndarray:
        return self.split_observation(observation).demand.astype(
            np.float32, copy=True
        )

    def append_observation(
        self,
        history: List[np.ndarray],
        observation: np.ndarray,
        current_period: int,
    ) -> None:
        if current_period <= 0:
            history.clear()
            return
        value = self.observed_demand(observation)
        if len(history) < current_period:
            history.append(value)
        elif history:
            history[-1] = value
