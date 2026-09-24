#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbit.utils import sha256_file  # noqa: E402
from scripts.merge_with_paper import aggregate  # noqa: E402


def main() -> int:
    errors = []
    lock = json.loads((ROOT / "vendor" / "LOCK.json").read_text())
    for relative, expected in lock["artifacts"].items():
        path = ROOT / "vendor" / relative
        if not path.exists() or sha256_file(path) != expected:
            errors.append(f"locked artifact mismatch: {relative}")

    for topology in ("base", "serial"):
        path = ROOT / "checkpoints" / f"fbit_{topology}_blind_final.pt"
        if not path.exists():
            errors.append(f"missing final checkpoint: {path.name}")
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("trainer_state", {})
        if int(state.get("global_step", 0)) < 200_000:
            errors.append(
                f"{path.name} has only {state.get('global_step', 0)} steps"
            )
        if payload.get("config", {}).get("topology") != topology:
            errors.append(f"{path.name} topology metadata mismatch")

    cache_path = ROOT / "results" / "cache_v2" / "bench_FBiT.csv"
    cache = None
    if not cache_path.exists():
        errors.append("missing bench_FBiT.csv")
    else:
        cache = pd.read_csv(cache_path)
        if len(cache) != 260:
            errors.append(f"FBiT cache has {len(cache)} rows, expected 260")
        if cache.duplicated(["ScenarioKey", "Seed"]).any():
            errors.append("FBiT cache has duplicate ScenarioKey/Seed rows")
        if cache["ScenarioKey"].nunique() != 26:
            errors.append("FBiT cache does not cover 26 scenarios")
        if cache["Seed"].nunique() != 10:
            errors.append("FBiT cache does not cover 10 canonical seeds")
        if "Failed" in cache and cache["Failed"].fillna(False).astype(bool).any():
            errors.append("FBiT cache contains failed episodes")
        for column in ("Profit", "FillRate", "AvgInv", "Unfulfilled", "Latency"):
            if column not in cache or not np.isfinite(cache[column]).all():
                errors.append(f"FBiT cache has missing/non-finite {column}")
        if "CheckpointSHA256" not in cache:
            errors.append("FBiT cache is missing full checkpoint SHA-256 provenance")
        else:
            for topology in ("base", "serial"):
                checkpoint = (
                    ROOT
                    / "checkpoints"
                    / f"fbit_{topology}_blind_final.pt"
                )
                rows = cache.loc[
                    cache["Network"].astype(str).eq(topology),
                    "CheckpointSHA256",
                ].dropna()
                expected_hash = (
                    sha256_file(checkpoint) if checkpoint.exists() else None
                )
                if (
                    expected_hash is None
                    or rows.nunique() != 1
                    or rows.iloc[0] != expected_hash
                ):
                    errors.append(
                        f"{topology} cache checkpoint SHA-256 mismatch"
                    )

    merged_path = ROOT / "results" / "benchmark_with_fbit.csv"
    if not merged_path.exists():
        errors.append("missing benchmark_with_fbit.csv")
    else:
        merged = pd.read_csv(merged_path)
        if len(merged) != 26:
            errors.append(f"merged result has {len(merged)} rows, expected 26")
        required = (
            "FBiT_Profit",
            "FBiT_Profit_Std",
            "FBiT_FillRate",
            "Time_Sec_FBiT",
            "FBiT_OptGap_Pct",
        )
        for column in required:
            if column not in merged or not np.isfinite(merged[column]).all():
                errors.append(f"merged result has missing/non-finite {column}")
        if cache is not None:
            try:
                expected = aggregate(cache, allow_partial=False).set_index(
                    "ScenarioKey"
                )
                actual = merged.set_index("ScenarioKey")
                for column in expected.columns:
                    if column not in actual:
                        errors.append(
                            f"merged result is missing recomputed {column}"
                        )
                        continue
                    left = expected[column].sort_index().to_numpy(dtype=float)
                    right = (
                        actual.loc[expected.index, column]
                        .sort_index()
                        .to_numpy(dtype=float)
                    )
                    if not np.allclose(
                        left, right, rtol=1e-10, atol=1e-10, equal_nan=True
                    ):
                        errors.append(
                            f"merged {column} cannot be reproduced from cache"
                        )
            except Exception as exc:
                errors.append(f"failed to recompute FBiT aggregates: {exc}")

    if errors:
        print("Verification failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Verification passed: checkpoints, 260-run cache, merged CSV, and locks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
