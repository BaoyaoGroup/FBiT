from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch_geometric.data import Batch

from .checkpoint import build_system_from_payload, load_payload
from .graph_adapter import CoreEnvGraphAdapter
from .utils import fbit_action_bounds, scaled_to_raw_action, sha256_file


class FBiTAgent:
    """Deterministic benchmark adapter with the paper's get_action interface."""

    def __init__(
        self,
        env,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ):
        self.env = env
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        payload = load_payload(self.checkpoint_path, self.device)
        self.system, self.config, self.metadata = build_system_from_payload(
            payload, self.device
        )
        self.system.eval()
        self.adapter = CoreEnvGraphAdapter(env, device="cpu")
        self.action_low, self.action_high = fbit_action_bounds(
            env,
            self.config.action_scaling_mode,
            self.config.raw_material_action_cap,
        )
        self._validate_metadata()
        self.history: List[np.ndarray] = []
        self.model_path = str(self.checkpoint_path)
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self.model_fingerprint = self.checkpoint_sha256[:12]

    def _validate_metadata(self) -> None:
        expected_topology = (
            "base" if len(self.adapter.reorder_links) == 11 else "serial"
        )
        if self.config.topology != expected_topology:
            raise ValueError(
                f"Checkpoint topology {self.config.topology} cannot control "
                f"{expected_topology} environment"
            )
        checks = {
            "num_nodes": self.adapter.num_nodes,
            "num_actions": self.adapter.num_actions,
            "num_retail": self.adapter.num_retail,
        }
        for key, expected in checks.items():
            if int(self.metadata[key]) != expected:
                raise ValueError(
                    f"Checkpoint {key}={self.metadata[key]}, environment={expected}"
                )
        expected_links = [list(edge) for edge in self.adapter.reorder_links]
        if self.metadata["reorder_links"] != expected_links:
            raise ValueError("Checkpoint action-edge ordering does not match CoreEnv")
        stored_high = self.metadata.get("action_scaling_high")
        if stored_high is not None and not np.array_equal(
            np.asarray(stored_high, dtype=np.float64), self.action_high
        ):
            raise ValueError(
                "Checkpoint action scaling does not match the environment"
            )

    def reset(self) -> None:
        self.history.clear()

    def get_action(
        self, obs: np.ndarray, current_period: int
    ) -> np.ndarray:
        self.adapter.append_observation(
            self.history, obs, current_period=current_period
        )
        graph = self.adapter.graph(obs, self.history)
        batch = Batch.from_data_list([graph]).to(self.device)
        with torch.no_grad():
            retail = self.system.predict_retail_demand(batch)
            predicted_nodes = self.system.demand_to_nodes(batch, retail)
            scaled, _ = self.system.actor(
                batch, predicted_nodes, deterministic=True
            )
        return scaled_to_raw_action(
            scaled.cpu().numpy(),
            self.action_low,
            self.action_high,
        )
