from __future__ import annotations

import gymnasium as gym
import numpy as np

from .config import ACTION_SCALING_EDGE_CAPS
from .utils import fbit_action_bounds


class EdgeCapRescaleAction(gym.ActionWrapper):
    """Expose ``[-1, 1]`` while sending FBiT-aligned quantities to CoreEnv.

    CoreEnv's own action space is left unchanged.  This wrapper is the training
    equivalent of FBiT's action conversion: distributor edges retain the
    environment high, factory edges use supplier capacity, and raw-material
    edges use ``raw_material_cap``.
    """

    def __init__(self, env: gym.Env, raw_material_cap: float = 100.0):
        super().__init__(env)
        self.raw_action_low, self.raw_action_high = fbit_action_bounds(
            env,
            ACTION_SCALING_EDGE_CAPS,
            raw_material_cap=raw_material_cap,
        )
        self.raw_material_cap = float(raw_material_cap)
        self.action_space = gym.spaces.Box(
            low=-np.ones_like(self.raw_action_low),
            high=np.ones_like(self.raw_action_high),
            dtype=np.float64,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        scaled = np.clip(
            np.asarray(action, dtype=np.float64), -1.0, 1.0
        )
        raw = self.raw_action_low + (scaled + 1.0) * 0.5 * (
            self.raw_action_high - self.raw_action_low
        )
        return np.clip(raw, self.raw_action_low, self.raw_action_high)

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        raw = np.clip(
            np.asarray(action, dtype=np.float64),
            self.raw_action_low,
            self.raw_action_high,
        )
        width = self.raw_action_high - self.raw_action_low
        return np.clip(
            2.0 * (raw - self.raw_action_low) / width - 1.0,
            -1.0,
            1.0,
        )
