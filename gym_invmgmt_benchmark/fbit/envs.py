from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from .paths import install_vendor_path

install_vendor_path()

from gym_invmgmt import CoreEnv  # noqa: E402


EFFECT_OPTIONS = ("trend", "seasonal", "shock")


class FullDomainRandomizationWrapper(gym.Wrapper):
    """Paper-compatible per-episode randomization without informed features."""

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        if seed is not None:
            # Seed CoreEnv first so its persistent generator is the source for
            # both domain selection and subsequent demand draws.
            self.env.reset(seed=seed, options=options)
        base_env = self.env.unwrapped
        rng = base_env.np_random

        effects = [effect for effect in EFFECT_OPTIONS if rng.random() < 0.5]
        external_series = None
        if rng.random() < 0.15:
            periods = int(base_env.num_periods)
            t = np.arange(periods)
            base = 20.0 + 0.02 * t * rng.uniform(0, 2)
            seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * t / 7)
            demand = rng.poisson(np.maximum(base * seasonal, 0.5))
            spikes = rng.random(periods) < 0.05
            demand[spikes] = (demand[spikes] * 3).astype(int)
            external_series = demand.astype(float)

        base_env.demand_engine.effects = effects
        base_env.demand_engine.type = "combined_chaos" if effects else "stationary"
        base_env.demand_engine.use_goodwill = bool(rng.choice([False, True]))
        base_env.demand_engine.noise_scale = float(rng.uniform(0.5, 2.0))
        base_env.demand_engine.sentiment = 1.0
        base_env.demand_engine.external_series = external_series
        base_env.backlog = bool(rng.choice([False, True]))
        return self.env.reset(options=options)


def make_training_env(topology: str, num_periods: int = 30) -> gym.Env:
    scenario = "network" if topology == "base" else "serial"
    env = CoreEnv(
        scenario=scenario,
        demand_config={"type": "stationary", "base_mu": 20},
        num_periods=num_periods,
        backlog=True,
    )
    return FullDomainRandomizationWrapper(env)


def make_stationary_env(topology: str, num_periods: int = 30) -> CoreEnv:
    scenario = "network" if topology == "base" else "serial"
    return CoreEnv(
        scenario=scenario,
        demand_config={
            "type": "stationary",
            "base_mu": 20,
            "use_goodwill": False,
        },
        num_periods=num_periods,
        backlog=True,
    )
