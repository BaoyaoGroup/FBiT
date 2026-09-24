#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbit.config import (  # noqa: E402
    ACTION_SCALING_EDGE_CAPS,
    ACTION_SCALING_MODES,
    FBiTConfig,
)
from fbit.trainer import FBiTTrainer  # noqa: E402
from fbit.utils import resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train topology-specific blind FBiT generalists."
    )
    parser.add_argument(
        "--topology", choices=("base", "serial", "all"), default="all"
    )
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:N")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the topology's resume checkpoint when present.",
    )
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--warmup-episodes", type=int, default=20)
    parser.add_argument("--updates-per-episode", type=int, default=10)
    parser.add_argument("--episode-batch-size", type=int, default=5)
    parser.add_argument(
        "--action-scaling",
        choices=ACTION_SCALING_MODES,
        default=ACTION_SCALING_EDGE_CAPS,
        help="edge_caps uses distributor/factory/raw-material caps per edge.",
    )
    parser.add_argument("--raw-material-cap", type=float, default=100.0)
    return parser.parse_args()


def train_one(args: argparse.Namespace, topology: str) -> Path:
    device = resolve_device(args.device)
    config = FBiTConfig(
        topology=topology,
        seed=args.seed,
        warmup_episodes=args.warmup_episodes,
        updates_per_episode=args.updates_per_episode,
        episode_batch_size=args.episode_batch_size,
        action_scaling_mode=args.action_scaling,
        raw_material_action_cap=args.raw_material_cap,
    )
    label = "base" if topology == "base" else "serial"
    trainer = FBiTTrainer(
        config=config,
        device=device,
        checkpoint_dir=ROOT / "checkpoints",
        log_dir=ROOT / "training_logs" / f"fbit_{label}_blind",
    )
    resume_path = trainer.resume_path if args.resume and trainer.resume_path.exists() else None
    print(
        f"Training FBiT topology={topology} steps={args.steps} "
        f"seed={args.seed} device={device} resume={resume_path} "
        f"action_scaling={config.action_scaling_mode} "
        f"action_high={trainer.action_high.tolist()}",
        flush=True,
    )
    return trainer.train(
        total_steps=args.steps,
        resume=resume_path,
        save_every=args.save_every,
    )


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.save_every < 0:
        raise SystemExit("--save-every cannot be negative")
    topologies = ("base", "serial") if args.topology == "all" else (args.topology,)
    for topology in topologies:
        path = train_one(args, topology)
        print(f"Final checkpoint: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
