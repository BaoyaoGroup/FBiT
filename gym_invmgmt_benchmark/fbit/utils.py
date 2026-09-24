from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import (
    ACTION_SCALING_EDGE_CAPS,
    ACTION_SCALING_ENVIRONMENT_HIGH,
)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(value: Optional[str] = None) -> torch.device:
    if value:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scaled_to_raw_action(
    scaled_action: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    scaled = np.clip(np.asarray(scaled_action, dtype=np.float64), -1.0, 1.0)
    raw = low + (scaled + 1.0) * 0.5 * (high - low)
    return np.clip(raw, low, high)


def fbit_action_bounds(
    env,
    mode: str,
    raw_material_cap: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return FBiT's action bounds without changing CoreEnv's action space.

    Under ``edge_caps``, distributor edges retain CoreEnv's upper bound,
    factory edges use the supplier's production capacity, and raw-material
    edges use ``raw_material_cap``.  Ordering follows ``reorder_links``.
    """
    low = np.asarray(env.action_space.low, dtype=np.float64).copy()
    environment_high = np.asarray(
        env.action_space.high, dtype=np.float64
    ).copy()
    if mode == ACTION_SCALING_ENVIRONMENT_HIGH:
        return low, environment_high
    if mode != ACTION_SCALING_EDGE_CAPS:
        raise ValueError(f"Unknown action scaling mode: {mode}")

    base_env = env.unwrapped
    network = base_env.network
    if len(network.reorder_links) != environment_high.size:
        raise ValueError("CoreEnv action space does not match reorder-link count")

    high = np.empty_like(environment_high)
    raw_material_nodes = set(network.rawmat)
    factory_nodes = set(network.factory)
    distributor_nodes = set(network.distrib)
    for index, (supplier, _purchaser) in enumerate(network.reorder_links):
        if supplier in raw_material_nodes:
            edge_cap = float(raw_material_cap)
        elif supplier in factory_nodes:
            edge_cap = float(network.graph.nodes[supplier]["C"])
        elif supplier in distributor_nodes:
            edge_cap = float(environment_high[index])
        else:
            raise ValueError(
                f"Cannot determine FBiT action cap for supplier {supplier!r}"
            )
        high[index] = min(edge_cap, float(environment_high[index]))

    if not np.isfinite(high).all() or (high <= low).any():
        raise ValueError("FBiT per-edge action bounds must be finite and positive")
    return low, high
