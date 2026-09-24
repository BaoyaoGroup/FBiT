from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict


ACTION_SCALING_EDGE_CAPS = "edge_caps"
ACTION_SCALING_ENVIRONMENT_HIGH = "environment_high"
ACTION_SCALING_MODES = (
    ACTION_SCALING_EDGE_CAPS,
    ACTION_SCALING_ENVIRONMENT_HIGH,
)


@dataclass
class FBiTConfig:
    """Serializable hyperparameters for one topology-specific FBiT model."""

    topology: str = "base"
    num_periods: int = 30
    node_feature_dim: int = 25
    edge_feature_dim: int = 5
    hidden_dim: int = 64
    graph_dim: int = 128
    graph_heads: int = 3
    predictor_dim: int = 64
    predictor_heads: int = 4
    predictor_layers: int = 3
    actor_hidden: int = 512
    critic_hidden: int = 1024
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    predictor_lr: float = 1e-3
    alpha_lr: float = 1e-3
    initial_alpha: float = 0.01
    reward_scale: float = 0.01
    replay_capacity_steps: int = 200_000
    episode_batch_size: int = 5
    warmup_episodes: int = 20
    updates_per_episode: int = 10
    action_scaling_mode: str = ACTION_SCALING_EDGE_CAPS
    raw_material_action_cap: float = 100.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.action_scaling_mode not in ACTION_SCALING_MODES:
            raise ValueError(
                f"Unknown action scaling mode: {self.action_scaling_mode}"
            )
        if (
            not math.isfinite(self.raw_material_action_cap)
            or self.raw_material_action_cap <= 0.0
        ):
            raise ValueError("raw_material_action_cap must be positive and finite")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "FBiTConfig":
        allowed = cls.__dataclass_fields__
        return cls(
            **{key: value for key, value in values.items() if key in allowed}
        )
