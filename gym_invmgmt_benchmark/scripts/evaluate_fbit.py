#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbit.agent import FBiTAgent  # noqa: E402
from fbit.paths import VENDOR_ROOT, install_vendor_path  # noqa: E402
from fbit.utils import sha256_file  # noqa: E402

install_vendor_path()
BENCHMARKS = VENDOR_ROOT / "benchmarks"
AGENTS = VENDOR_ROOT / "agents"
for path in (BENCHMARKS, AGENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.run_benchmarks import (  # noqa: E402
    CANONICAL_SEEDS,
    _env_kwargs,
    _extract_kpis,
    build_all_scenarios,
    scenario_id,
)
from gym_invmgmt import CoreEnv  # noqa: E402


CACHE_PATH = ROOT / "results" / "cache_v2" / "bench_FBiT.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FBiT under the locked paper benchmark contract."
    )
    parser.add_argument(
        "--scenarios",
        choices=("all", "core", "marl"),
        default="all",
        help="all=26, core=22 non-MARL, marl=4 supplemental rows",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=ROOT / "checkpoints"
    )
    parser.add_argument(
        "--checkpoint-kind",
        choices=("final", "best", "resume"),
        default="final",
        help="Checkpoint snapshot to evaluate (default: final).",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Result cache path; defaults to a snapshot-specific CSV.",
    )
    return parser.parse_args()


def checkpoint_for(
    checkpoint_dir: Path, network: str, kind: str = "final"
) -> Path:
    label = "base" if network == "base" else "serial"
    return checkpoint_dir / f"fbit_{label}_blind_{kind}.pt"


def cache_for(kind: str) -> Path:
    if kind == "final":
        return CACHE_PATH
    return ROOT / "results" / "cache_v2" / f"bench_FBiT_{kind}.csv"


def load_completed(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if "CheckpointSHA256" not in frame.columns:
        return set()
    if "Failed" in frame.columns:
        frame = frame[frame["Failed"] != True]  # noqa: E712
    frame = frame.dropna(subset=["Profit"])
    return {
        (
            str(row.ScenarioKey),
            int(row.Seed),
            str(row.CheckpointSHA256),
        )
        for row in frame.itertuples(index=False)
    }


def append_row(path: Path, row: Dict[str, object]) -> None:
    """Atomically upsert one scenario/seed result.

    Successful cache entries are skipped by ``load_completed``.  Failed or
    otherwise incomplete entries are replaced here when they are retried, so
    interruption/retry cannot create duplicate scenario/seed rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        same_key = (
            old["ScenarioKey"].astype(str).eq(str(row["ScenarioKey"]))
            & old["Seed"].astype(int).eq(int(row["Seed"]))
        )
        old = old.loc[~same_key]
        columns = list(old.columns) + [
            column for column in new.columns if column not in old.columns
        ]
        output = pd.concat(
            (old.reindex(columns=columns), new.reindex(columns=columns)),
            ignore_index=True,
        )
    else:
        output = new
    temporary = path.with_suffix(path.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(path)


def run_episode(env: CoreEnv, agent: FBiTAgent, seed: int) -> Dict[str, float]:
    observation, _ = env.reset(seed=seed)
    agent.reset()
    period = 0
    done = False
    started = time.perf_counter()
    while not done:
        action = agent.get_action(observation, period)
        observation, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        period += 1
    replay_time = time.perf_counter() - started
    result = _extract_kpis(env)
    result["PlanTime"] = 0.0
    result["ReplayTime"] = replay_time
    result["Latency"] = replay_time
    return result


def selected_scenarios(mode: str) -> List[dict]:
    scenarios = build_all_scenarios()
    if mode == "core":
        return [item for item in scenarios if not item["marl"]]
    if mode == "marl":
        return [item for item in scenarios if item["marl"]]
    return scenarios


def main() -> int:
    args = parse_args()
    cache_path = args.cache_path or cache_for(args.checkpoint_kind)
    if args.seeds <= 0:
        raise SystemExit("--seeds must be positive")
    if args.seeds > len(CANONICAL_SEEDS):
        raise SystemExit(
            f"--seeds cannot exceed the {len(CANONICAL_SEEDS)} canonical seeds"
        )
    scenarios = selected_scenarios(args.scenarios)
    selected_seeds = CANONICAL_SEEDS[: args.seeds]
    if args.force and cache_path.exists():
        current = pd.read_csv(cache_path)
        keys = {scenario_id(item) for item in scenarios}
        current = current[~current["ScenarioKey"].isin(keys)]
        if current.empty:
            cache_path.unlink()
        else:
            current.to_csv(cache_path, index=False)
    completed = load_completed(cache_path)
    checkpoint_hashes: Dict[Path, str] = {}
    total = len(scenarios) * len(selected_seeds)
    progress = 0
    for scenario in scenarios:
        sid = scenario_id(scenario)
        checkpoint = checkpoint_for(
            args.checkpoint_dir, scenario["network"], args.checkpoint_kind
        )
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing {checkpoint}; train both topology checkpoints first"
            )
        if checkpoint not in checkpoint_hashes:
            checkpoint_hashes[checkpoint] = sha256_file(checkpoint)
        checkpoint_hash = checkpoint_hashes[checkpoint]
        for seed in selected_seeds:
            progress += 1
            if (sid, seed, checkpoint_hash) in completed:
                print(f"[{progress}/{total}] cached {sid} seed={seed}")
                continue
            env = CoreEnv(**_env_kwargs(scenario))
            init_started = time.perf_counter()
            agent = FBiTAgent(env, checkpoint, device=args.device)
            plan_time = time.perf_counter() - init_started
            try:
                kpis = run_episode(env, agent, seed)
                kpis["PlanTime"] = plan_time
                kpis["Latency"] = plan_time + kpis["ReplayTime"]
                failed = False
                reason = ""
            except Exception as exc:
                kpis = {
                    key: np.nan
                    for key in (
                        "Profit",
                        "FillRate",
                        "AvgInv",
                        "Unfulfilled",
                        "Latency",
                        "PlanTime",
                        "ReplayTime",
                    )
                }
                failed = True
                reason = f"{type(exc).__name__}: {exc}"
            finally:
                env.close()
            row: Dict[str, object] = {
                "ScenarioKey": sid,
                "Block": scenario["block"],
                "Network": scenario["network"],
                "Demand": scenario["demand_label"],
                "Goodwill": scenario["use_goodwill"],
                "Backlog": scenario["backlog"],
                "MARL": scenario["marl"],
                "Seed": seed,
                "CheckpointKind": args.checkpoint_kind,
                "ModelFingerprint": agent.model_fingerprint,
                "CheckpointSHA256": agent.checkpoint_sha256,
                "ActionScalingMode": agent.config.action_scaling_mode,
                "RawMaterialActionCap": agent.config.raw_material_action_cap,
                "ActionHigh": json.dumps(agent.action_high.tolist()),
                "ModelPath": str(checkpoint),
                "Failed": failed,
                "FailReason": reason,
                **kpis,
            }
            append_row(cache_path, row)
            print(
                f"[{progress}/{total}] {sid} seed={seed} "
                f"profit={row['Profit']} failed={failed}",
                flush=True,
            )
    print(f"FBiT cache: {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
