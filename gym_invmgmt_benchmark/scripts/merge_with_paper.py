#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbit.paths import VENDOR_ROOT  # noqa: E402
from fbit.utils import sha256_file  # noqa: E402


OFFICIAL = VENDOR_ROOT / "results" / "benchmark_final_merged.csv"
FBIT_CACHE = ROOT / "results" / "cache_v2" / "bench_FBiT.csv"
OUTPUT = ROOT / "results" / "benchmark_with_fbit.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge FBiT results beside the immutable official artifact."
    )
    parser.add_argument("--official", type=Path, default=OFFICIAL)
    parser.add_argument("--fbit-cache", type=Path, default=FBIT_CACHE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow fewer than ten successful seeds per scenario.",
    )
    return parser.parse_args()


def aggregate(cache: pd.DataFrame, allow_partial: bool) -> pd.DataFrame:
    if "Failed" in cache.columns:
        cache = cache[cache["Failed"] != True]  # noqa: E712
    cache = cache.dropna(subset=["Profit"])
    if cache.duplicated(["ScenarioKey", "Seed"]).any():
        raise ValueError("FBiT cache contains duplicate ScenarioKey/Seed rows")
    counts = cache.groupby("ScenarioKey")["Seed"].nunique()
    if not allow_partial and (
        len(counts) != 26 or (counts != 10).any()
    ):
        raise ValueError(
            "Complete merge requires 26 scenarios with exactly 10 successful seeds"
        )
    rows = []
    for scenario_key, group in cache.groupby("ScenarioKey", sort=False):
        profits = group["Profit"].to_numpy(dtype=float)
        row = {
            "ScenarioKey": scenario_key,
            "FBiT_Profit": float(profits.mean()),
            "FBiT_Profit_Std": float(profits.std(ddof=0)),
            "FBiT_CVaR5": float(np.sort(profits)[: max(1, int(0.05 * len(profits)))].mean()),
        }
        for source, target in (
            ("FillRate", "FBiT_FillRate"),
            ("AvgInv", "FBiT_AvgInv"),
            ("Unfulfilled", "FBiT_Unfulfilled"),
            ("Revenue", "FBiT_Revenue"),
            ("HoldingCost", "FBiT_HoldingCost"),
            ("BacklogPenalty", "FBiT_BacklogPenalty"),
            ("ProcurementCost", "FBiT_ProcurementCost"),
            ("OperatingCost", "FBiT_OperatingCost"),
            ("FixedOrderingCost", "FBiT_FixedOrderingCost"),
            ("BullwhipRatio", "FBiT_BullwhipRatio"),
            ("ServiceLevel", "FBiT_ServiceLevel"),
            ("Latency", "Time_FBiT"),
            ("PlanTime", "FBiT_PlanTime"),
            ("ReplayTime", "FBiT_ReplayTime"),
        ):
            if source in group.columns:
                row[target] = float(group[source].mean())
        row["Time_Sec_FBiT"] = row.get("Time_FBiT", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.output.resolve() == args.official.resolve():
        raise ValueError("Output must not overwrite the official benchmark artifact")
    official_hash = sha256_file(args.official)
    official = pd.read_csv(args.official)
    fbit = aggregate(pd.read_csv(args.fbit_cache), args.allow_partial)
    merged = official.merge(
        fbit, on="ScenarioKey", how="left", validate="one_to_one"
    )
    if not args.allow_partial and merged["FBiT_Profit"].isna().any():
        raise ValueError("Some official scenarios have no FBiT aggregate")
    merged["FBiT_OptGap_Pct"] = (
        (merged["Oracle_Profit"] - merged["FBiT_Profit"])
        / merged["Oracle_Profit"].abs()
        * 100.0
    ).round(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    if sha256_file(args.official) != official_hash:
        raise RuntimeError("Official benchmark artifact changed during merge")
    print(
        f"Merged {len(merged)} scenarios to {args.output}; "
        f"official sha256={official_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
