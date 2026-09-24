#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbit.config import ACTION_SCALING_EDGE_CAPS  # noqa: E402
from fbit.paths import VENDOR_ROOT, install_vendor_path  # noqa: E402
from fbit.utils import fbit_action_bounds, sha256_file  # noqa: E402

install_vendor_path()
for path in (VENDOR_ROOT / "benchmarks", VENDOR_ROOT / "agents"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agents.dagger_agent import DAggerAgent  # noqa: E402
from agents.ppo_baseline_agent import RLAgent  # noqa: E402
from agents.ppo_gnn_agent import PPOGNNAgent  # noqa: E402
from agents.sac_eval_agent import SACAgent  # noqa: E402
from agents.st_ppo_agent import STPPOAgent  # noqa: E402
from benchmarks.run_benchmarks import (  # noqa: E402
    CANONICAL_SEEDS,
    _env_kwargs,
    _extract_kpis,
    build_all_scenarios,
    scenario_id,
)
from gym_invmgmt import CoreEnv  # noqa: E402


RETRAINABLE_METHODS = (
    "PPO-MLP-B",
    "PPO-GNN-B",
    "ST-PPO-B",
    "SAC-B",
)
EXTERNAL_ARTIFACT_METHODS = (
    "DAgger-B",
)
METHODS = RETRAINABLE_METHODS + EXTERNAL_ARTIFACT_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrained Blind RL models with FBiT edge caps."
    )
    parser.add_argument("--method", choices=(*METHODS, "all"), required=True)
    parser.add_argument(
        "--scenarios", choices=("all", "core", "marl"), default="all"
    )
    parser.add_argument(
        "--topology", choices=("base", "serial", "all"), default="all"
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--model-kind", choices=("final", "best"), default="final")
    parser.add_argument("--raw-material-cap", type=float, default=100.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "blind_edge_caps"
    )
    parser.add_argument(
        "--result-dir", type=Path, default=ROOT / "results" / "blind_edge_caps"
    )
    return parser.parse_args()


def selected_scenarios(mode: str, topology: str) -> list[dict]:
    scenarios = build_all_scenarios()
    if topology != "all":
        scenarios = [
            scenario for scenario in scenarios if scenario["network"] == topology
        ]
    if mode == "core":
        return [scenario for scenario in scenarios if not scenario["marl"]]
    if mode == "marl":
        return [scenario for scenario in scenarios if scenario["marl"]]
    return scenarios


def artifact_paths(
    checkpoint_dir: Path,
    method: str,
    topology: str,
    kind: str,
) -> tuple[Path, Path, Path]:
    run_dir = checkpoint_dir / method / topology
    if kind == "best":
        return (
            run_dir / "best_model.zip",
            run_dir / "best_vecnormalize.pkl",
            run_dir / "training_config.json",
        )
    return (
        run_dir / "final_model.zip",
        run_dir / "vecnormalize.pkl",
        run_dir / "training_config.json",
    )


def build_agent(
    method: str,
    env,
    model: Path,
    stats: Path,
    raw_material_cap: float,
):
    if method == "PPO-MLP-B":
        agent = RLAgent(
            env,
            model_path=str(model),
            stats_path=str(stats),
            obs_level="v2",
            is_blind=True,
            verbose=False,
        )
    elif method == "PPO-GNN-B":
        agent = PPOGNNAgent(
            env,
            model_path=str(model),
            stats_path=str(stats),
            is_blind=True,
        )
    elif method == "ST-PPO-B":
        agent = STPPOAgent(
            env,
            model_path=str(model),
            stats_path=str(stats),
            is_blind=True,
        )
    elif method == "SAC-B":
        agent = SACAgent(
            env,
            model_path=str(model),
            stats_path=str(stats),
            is_blind=True,
        )
    elif method == "DAgger-B":
        agent = DAggerAgent(
            env,
            specialist=False,
            model_path=str(model),
            stats_path=str(stats),
            is_blind=True,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    action_low, action_high = fbit_action_bounds(
        env,
        ACTION_SCALING_EDGE_CAPS,
        raw_material_cap=raw_material_cap,
    )
    agent.action_low = action_low
    agent.action_high = action_high
    return agent, action_low, action_high


def append_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        same = (
            old["ScenarioKey"].astype(str).eq(str(row["ScenarioKey"]))
            & old["Seed"].astype(int).eq(int(row["Seed"]))
        )
        old = old.loc[~same]
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


def load_completed(path: Path, model_hashes: set[str]) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    required = {"ScenarioKey", "Seed", "ModelSHA256", "Failed", "Profit"}
    if not required.issubset(frame.columns):
        return set()
    frame = frame[
        (frame["Failed"] != True)  # noqa: E712
        & frame["Profit"].notna()
        & frame["ModelSHA256"].isin(model_hashes)
    ]
    return {
        (str(row.ScenarioKey), int(row.Seed), str(row.ModelSHA256))
        for row in frame.itertuples(index=False)
    }


def run_episode(env, agent, action_high: np.ndarray, seed: int):
    observation, _ = env.reset(seed=seed)
    period = 0
    done = False
    started = time.perf_counter()
    max_violation = 0.0
    while not done:
        action = np.asarray(agent.get_action(observation, period), dtype=np.float64)
        if not np.isfinite(action).all():
            raise FloatingPointError("Agent produced a non-finite action")
        max_violation = max(
            max_violation, float(np.max(np.maximum(action - action_high, 0.0)))
        )
        observation, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        period += 1
    replay_time = time.perf_counter() - started
    kpis = _extract_kpis(env)
    kpis["ReplayTime"] = replay_time
    kpis["TotalRequested"] = float(np.sum(env.action_log))
    kpis["TotalExecuted"] = float(np.sum(env.R))
    kpis["MaxActionBoundViolation"] = max_violation
    return kpis


def evaluate_method(args: argparse.Namespace, method: str) -> Path:
    scenarios = selected_scenarios(args.scenarios, args.topology)
    seeds = CANONICAL_SEEDS[: args.seeds]
    cache_path = args.result_dir / f"bench_{method}_{args.model_kind}.csv"
    artifacts = {}
    hashes = set()
    required_topologies = sorted({scenario["network"] for scenario in scenarios})
    for topology in required_topologies:
        model, stats, config = artifact_paths(
            args.checkpoint_dir, method, topology, args.model_kind
        )
        for path in (model, stats, config):
            if not path.exists():
                raise FileNotFoundError(f"Missing retrained artifact: {path}")
        metadata = json.loads(config.read_text(encoding="utf-8"))
        if metadata.get("action_scaling") != "edge_caps":
            raise ValueError(f"{config} is not an edge-caps training run")
        if float(metadata.get("raw_material_cap")) != args.raw_material_cap:
            raise ValueError(
                f"{config} raw-material cap does not match evaluation"
            )
        model_hash = sha256_file(model)
        hashes.add(model_hash)
        artifacts[topology] = (model, stats, model_hash)
    if args.force and cache_path.exists():
        cache_path.unlink()
    completed = load_completed(cache_path, hashes)

    total = len(scenarios) * len(seeds)
    progress = 0
    for scenario in scenarios:
        topology = scenario["network"]
        model, stats, model_hash = artifacts[topology]
        sid = scenario_id(scenario)
        for seed in seeds:
            progress += 1
            if (sid, seed, model_hash) in completed:
                print(f"[{progress}/{total}] cached {method} {sid} seed={seed}")
                continue
            env = CoreEnv(**_env_kwargs(scenario))
            plan_started = time.perf_counter()
            agent, _action_low, action_high = build_agent(
                method,
                env,
                model,
                stats,
                args.raw_material_cap,
            )
            plan_time = time.perf_counter() - plan_started
            try:
                kpis = run_episode(env, agent, action_high, seed)
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
                        "ReplayTime",
                        "TotalRequested",
                        "TotalExecuted",
                        "MaxActionBoundViolation",
                    )
                }
                failed = True
                reason = f"{type(exc).__name__}: {exc}"
            finally:
                env.close()
            row = {
                "Method": method,
                "ScenarioKey": sid,
                "Block": scenario["block"],
                "Network": topology,
                "Demand": scenario["demand_label"],
                "Goodwill": scenario["use_goodwill"],
                "Backlog": scenario["backlog"],
                "MARL": scenario["marl"],
                "Seed": seed,
                "ModelKind": args.model_kind,
                "ModelPath": str(model),
                "ModelSHA256": model_hash,
                "ActionScaling": "edge_caps",
                "ActionHigh": json.dumps(action_high.tolist()),
                "PlanTime": plan_time,
                "Failed": failed,
                "FailReason": reason,
                **kpis,
            }
            append_row(cache_path, row)
            print(
                f"[{progress}/{total}] {method} {sid} seed={seed} "
                f"profit={row['Profit']} failed={failed}",
                flush=True,
            )
    return cache_path


def main() -> int:
    args = parse_args()
    if args.seeds <= 0 or args.seeds > len(CANONICAL_SEEDS):
        raise SystemExit(f"--seeds must be in [1, {len(CANONICAL_SEEDS)}]")
    # The upstream release does not contain the DAgger-B training program.
    # Keep explicit DAgger evaluation available for artifacts produced by an
    # author-supplied trainer, but do not imply that `all` can retrain/evaluate
    # it from the public release alone.
    methods = RETRAINABLE_METHODS if args.method == "all" else (args.method,)
    for method in methods:
        path = evaluate_method(args, method)
        print(f"saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
