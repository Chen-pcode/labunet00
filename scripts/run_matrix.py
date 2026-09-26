"""Print an experiment matrix by default; --execute runs jobs serially on one GPU."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--sources", nargs="+", default=["isic2018"], choices=["isic2017", "isic2018"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--variants", nargs="+", default=["baseline", "unfused_control", "sampling_only", "constant_scale", "geometry"])
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Final test evaluation; omit during validation-only screening")
    args = parser.parse_args()
    for source in args.sources:
        for variant in args.variants:
            config = ROOT / "configs" / "ablations" / f"{variant}.yaml"
            if not config.is_file():
                raise ValueError(f"Unknown configuration: {variant}")
            for seed in args.seeds:
                run_dir = Path(args.output_root).resolve() / source / variant / f"seed_{seed}"
                command = [sys.executable, "-m", "skinmamba", "run" if args.evaluate else "train",
                    "--config", str(config), "--data-root", str(Path(args.data_root).resolve()),
                    "--run-dir", str(run_dir), "--set", f"data.source={source}", "--set", f"seed={seed}"]
                for override in args.overrides:
                    command.extend(["--set", override])
                print(subprocess.list2cmdline(command), flush=True)
                if args.execute:
                    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
