#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbit.edge_cap_wrapper import EdgeCapRescaleAction  # noqa: E402
from fbit.paths import VENDOR_ROOT, install_vendor_path  # noqa: E402

install_vendor_path()
for path in (VENDOR_ROOT / "benchmarks", VENDOR_ROOT / "agents"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.train_generalist import (  # noqa: E402
    FullDomainRandomizationWrapper,
    build_model,
)
from gym_invmgmt import CoreEnv  # noqa: E402
from gym_invmgmt.wrappers.domain_features import (  # noqa: E402
    DomainFeatureWrapper,
)


METHOD_TO_ARCH = {
    "PPO-MLP-B": "ppo-mlp",
    "PPO-GNN-B": "ppo-gnn",
    "ST-PPO-B": "st-ppo",
    "SAC-B": "sac",
}


class SaveVecNormalizeOnBest(BaseCallback):
    def __init__(self, path: Path):
        super().__init__(verbose=0)
        self.path = path

    def _on_step(self) -> bool:
        vecnormalize = self.model.get_vec_normalize_env()
        if vecnormalize is None:
            raise RuntimeError("Training model has no VecNormalize environment")
        with self.path.open("wb") as handle:
            pickle.dump(vecnormalize, handle)
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain Blind RL models with FBiT-aligned per-edge action caps."
        )
    )
    parser.add_argument(
        "--method", choices=(*METHOD_TO_ARCH, "all"), required=True
    )
    parser.add_argument(
        "--topology", choices=("base", "serial", "all"), required=True
    )
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-material-cap", type=float, default=100.0)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "checkpoints" / "blind_edge_caps"
    )
    parser.add_argument(
        "--log-dir", type=Path, default=ROOT / "training_logs" / "blind_edge_caps"
    )
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    return parser.parse_args()


def make_env(
    arch: str,
    topology: str,
    raw_material_cap: float,
    randomized: bool,
):
    scenario = "network" if topology == "base" else "serial"

    def initialize():
        env = CoreEnv(
            scenario=scenario,
            demand_config={
                "type": "stationary",
                "base_mu": 20,
                "use_goodwill": False,
            },
            num_periods=30,
            backlog=True,
        )
        if randomized:
            env = FullDomainRandomizationWrapper(env)
        env = DomainFeatureWrapper(
            env, is_blind=True, enhanced=True, grouped=True
        )
        if arch == "st-ppo":
            from gym_invmgmt.wrappers.temporal_frame_stack import (
                TemporalFrameStack,
            )

            env = TemporalFrameStack(env, n_history=4)
        env = EdgeCapRescaleAction(
            env, raw_material_cap=raw_material_cap
        )
        return Monitor(env)

    return initialize


def train_one(args: argparse.Namespace, method: str, topology: str) -> Path:
    arch = METHOD_TO_ARCH[method]
    set_random_seed(args.seed)
    run_dir = args.output_dir / method / topology
    tensorboard_dir = args.log_dir / method / topology
    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv(
        [
            make_env(
                arch,
                topology,
                args.raw_material_cap,
                randomized=True,
            )
        ]
    )
    train_env.seed(args.seed)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=(arch != "sac"),
        clip_obs=10.0,
        gamma=0.995,
    )

    eval_env = DummyVecEnv(
        [
            make_env(
                arch,
                topology,
                args.raw_material_cap,
                randomized=False,
            )
        ]
    )
    eval_env.seed(args.seed + 99)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.995,
        training=False,
    )
    eval_env.obs_rms = train_env.obs_rms

    scenario = "network" if topology == "base" else "serial"
    model = build_model(
        arch, train_env, str(tensorboard_dir), topology=scenario
    )
    model.set_random_seed(args.seed)
    wrapped_env = train_env.envs[0]
    raw_high = wrapped_env.unwrapped.action_space.high.tolist()
    edge_high = wrapped_env.get_wrapper_attr("raw_action_high").tolist()
    metadata = {
        "method": method,
        "architecture": arch,
        "topology": topology,
        "steps": args.steps,
        "seed": args.seed,
        "action_scaling": "edge_caps",
        "raw_material_cap": args.raw_material_cap,
        "environment_action_high": raw_high,
        "edge_action_high": edge_high,
    }
    (run_dir / "training_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)

    callbacks = []
    if args.eval_freq > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                callback_on_new_best=SaveVecNormalizeOnBest(
                    run_dir / "best_vecnormalize.pkl"
                ),
                best_model_save_path=str(run_dir),
                log_path=str(run_dir),
                eval_freq=args.eval_freq,
                n_eval_episodes=5,
                deterministic=True,
                verbose=1,
            )
        )
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(run_dir / "training_checkpoints"),
                name_prefix=f"{method}_{topology}",
                save_replay_buffer=(arch == "sac"),
                save_vecnormalize=True,
            )
        )

    model.learn(
        total_timesteps=args.steps,
        callback=CallbackList(callbacks),
    )
    final_path = run_dir / "final_model"
    model.save(final_path)
    with (run_dir / "vecnormalize.pkl").open("wb") as handle:
        pickle.dump(train_env, handle)
    train_env.close()
    eval_env.close()
    print(f"saved {final_path}.zip", flush=True)
    return final_path.with_suffix(".zip")


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.raw_material_cap <= 0:
        raise SystemExit("--raw-material-cap must be positive")
    methods = tuple(METHOD_TO_ARCH) if args.method == "all" else (args.method,)
    topologies = (
        ("base", "serial") if args.topology == "all" else (args.topology,)
    )
    for method in methods:
        for topology in topologies:
            train_one(args, method, topology)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
