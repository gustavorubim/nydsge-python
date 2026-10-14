"""Repository-local CLI with an owned CPU subprocess and a hard compute timeout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Compute a unique offline archive (no plotting)")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--timeout-seconds", type=int, default=600)
    plot = commands.add_parser("plot", help="Render a completed archive without solving")
    plot.add_argument("--run-dir", type=Path, required=True)
    plot.add_argument("--output-dir", type=Path)
    worker = commands.add_parser("_compute", help=argparse.SUPPRESS)
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    if args.command == "run":
        if not 1 <= args.timeout_seconds <= 1800:
            parser.error("timeout-seconds must be 1–1800")
        out = args.output_dir or Path("outputs/general_analytics_tools") / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        )
        if out.exists():
            parser.error("Output already exists; use a new directory to preserve provenance.")
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "general_analytics_tools",
                    "_compute",
                    "--config",
                    str(args.config.resolve()),
                    "--output-dir",
                    str(out.resolve()),
                ],
                env=os.environ.copy(),
                check=True,
                timeout=args.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            if out.is_dir():
                (out / "status.json").write_text(
                    json.dumps({"status": "timed_out"}), encoding="utf-8"
                )
            raise SystemExit(
                "Own compute subprocess timed out; no complete result or plots."
            ) from error
        print(f"Completed archive: {out.resolve()}")
    elif args.command == "_compute":
        from general_analytics_tools.compute import compute

        result = compute(args.config, args.output_dir)
        print(json.dumps({"forecast_start": result["forecast_start"], "HD": result["HD"]}))
    else:
        from general_analytics_tools.plotting import render

        for path in render(args.run_dir, args.output_dir):
            print(path.resolve())


if __name__ == "__main__":
    main()
