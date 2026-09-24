#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train both FBiT generalists, evaluate 26 scenarios, and merge."
    )
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--devices",
        default="auto",
        help="Comma-separated physical GPU IDs, auto, or cpu.",
    )
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only from a checkpoint with matching action scaling.",
    )
    return parser.parse_args()


def available_gpu_ids() -> List[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, free = [part.strip() for part in line.split(",", maxsplit=1)]
        rows.append((int(free), index))
    return [index for _free, index in sorted(rows, reverse=True)]


def command_for_training(
    topology: str, steps: int, seed: int, resume: bool = False
) -> List[str]:
    command = [
        str(PYTHON),
        str(ROOT / "scripts" / "train_fbit.py"),
        "--topology",
        topology,
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--device",
        "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    ]
    if resume:
        command.append("--resume")
    return command


def run_training(args: argparse.Namespace) -> None:
    if args.devices == "cpu":
        devices: List[str] = []
    elif args.devices == "auto":
        devices = available_gpu_ids()[:2]
    else:
        devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    parallel = len(devices) >= 2 and not args.sequential
    jobs = []
    logs = ROOT / "training_logs"
    logs.mkdir(parents=True, exist_ok=True)
    for idx, topology in enumerate(("base", "serial")):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            (str(ROOT), env.get("PYTHONPATH", ""))
        )
        env["MPLCONFIGDIR"] = str(logs / "matplotlib")
        if devices:
            env["CUDA_VISIBLE_DEVICES"] = devices[idx % len(devices)]
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        command = [
            str(PYTHON),
            str(ROOT / "scripts" / "train_fbit.py"),
            "--topology",
            topology,
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--device",
            "cuda" if devices else "cpu",
        ]
        if args.resume:
            command.append("--resume")
        log_path = logs / f"launcher_{topology}.log"
        if parallel:
            handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            jobs.append((topology, process, handle, log_path))
            print(
                f"started {topology} pid={process.pid} "
                f"gpu={env['CUDA_VISIBLE_DEVICES']} log={log_path}"
            )
        else:
            print(
                f"running {topology} gpu={env['CUDA_VISIBLE_DEVICES'] or 'CPU'} "
                f"log={log_path}"
            )
            with log_path.open("a", encoding="utf-8") as handle:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
    failures = []
    for topology, process, handle, log_path in jobs:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append(f"{topology} exit={return_code} log={log_path}")
    if failures:
        raise RuntimeError("Training failed: " + "; ".join(failures))


def run_checked(command: List[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if not args.skip_train:
        run_training(args)
    if not args.skip_eval:
        command = [
            str(PYTHON),
            str(ROOT / "scripts" / "evaluate_fbit.py"),
            "--scenarios",
            "all",
            "--seeds",
            "10",
        ]
        if args.force_eval:
            command.append("--force")
        run_checked(command)
    run_checked(
        [str(PYTHON), str(ROOT / "scripts" / "merge_with_paper.py")]
    )
    run_checked(
        [str(PYTHON), str(ROOT / "scripts" / "verify_results.py")]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
