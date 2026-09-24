from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fbit.paths import PROJECT_ROOT, VENDOR_ROOT
from fbit.utils import sha256_file
from scripts.evaluate_fbit import append_row
from scripts.merge_with_paper import aggregate


def test_locked_official_artifacts_are_unchanged():
    lock = json.loads((PROJECT_ROOT / "vendor" / "LOCK.json").read_text())
    for relative_path, expected_hash in lock["artifacts"].items():
        path = PROJECT_ROOT / "vendor" / relative_path
        assert path.exists()
        assert sha256_file(path) == expected_hash


def test_official_benchmark_contract():
    path = VENDOR_ROOT / "results" / "benchmark_final_merged.csv"
    frame = pd.read_csv(path)
    assert len(frame) == 26
    assert (~frame["MARL"]).sum() == 22
    assert frame["MARL"].sum() == 4
    assert "FBiT_Profit" not in frame.columns


def test_result_cache_upserts_failed_retry_atomically(tmp_path):
    cache = tmp_path / "bench_FBiT.csv"
    common = {
        "ScenarioKey": "scenario",
        "Seed": 42,
        "Network": "base",
    }
    append_row(cache, {**common, "Failed": True, "Profit": np.nan})
    append_row(cache, {**common, "Failed": False, "Profit": 123.0})
    frame = pd.read_csv(cache)
    assert len(frame) == 1
    assert frame.iloc[0]["Profit"] == 123.0
    assert not cache.with_suffix(".csv.tmp").exists()


def test_aggregate_values_are_computed_from_seed_rows():
    cache = pd.DataFrame(
        {
            "ScenarioKey": ["one", "one"],
            "Seed": [42, 123],
            "Failed": [False, False],
            "Profit": [10.0, 30.0],
            "FillRate": [0.5, 0.9],
            "Latency": [1.0, 3.0],
        }
    )
    row = aggregate(cache, allow_partial=True).iloc[0]
    assert row["FBiT_Profit"] == 20.0
    assert row["FBiT_Profit_Std"] == 10.0
    assert row["FBiT_CVaR5"] == 10.0
    assert row["FBiT_FillRate"] == 0.7
    assert row["Time_Sec_FBiT"] == 2.0
