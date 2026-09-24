from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .config import FBiTConfig
from .networks import FBiTSystem


CHECKPOINT_VERSION = 5


def model_metadata(adapter) -> Dict[str, Any]:
    return {
        "num_nodes": adapter.num_nodes,
        "num_actions": adapter.num_actions,
        "num_retail": adapter.num_retail,
        "node_feature_dim": adapter.node_feature_dim,
        "pipeline_arrival_horizon": adapter.pipeline_arrival_horizon,
        "retailer_node_indices": [
            adapter.node_to_local[edge[0]] for edge in adapter.retail_links
        ],
        "reorder_links": [list(edge) for edge in adapter.reorder_links],
    }


def save_checkpoint(
    path: Path,
    system: FBiTSystem,
    config: FBiTConfig,
    metadata: Dict[str, Any],
    trainer_state: Optional[Dict[str, Any]] = None,
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    replay_state: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "config": config.to_dict(),
        "metadata": metadata,
        "model": system.state_dict(),
        "trainer_state": trainer_state or {},
    }
    if optimizers is not None:
        payload["optimizers"] = {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        }
    if replay_state is not None:
        payload["replay"] = replay_state
    payload["rng"] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_payload(path: Path, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def build_system_from_payload(
    payload: Dict[str, Any], device: torch.device
) -> tuple[FBiTSystem, FBiTConfig, Dict[str, Any]]:
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported FBiT checkpoint version: {payload.get('checkpoint_version')}"
        )
    config = FBiTConfig.from_dict(payload["config"])
    metadata = dict(payload["metadata"])
    system = FBiTSystem(
        config,
        num_retail=int(metadata["num_retail"]),
        retailer_node_indices=metadata["retailer_node_indices"],
        num_nodes=int(metadata["num_nodes"]),
        num_actions=int(metadata["num_actions"]),
    ).to(device)
    system.load_state_dict(payload["model"])
    return system, config, metadata
