"""Print an experiment matrix by default; --execute runs jobs serially on one GPU."""
from __future__ import annotations

import argparse
import subprocess
import sys
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skinmamba.config import ABLATIONS

STAGES = {
    "core": ["baseline", "hsm_only", "hsm_local", "readback"],
    "bridge": ["readback", "readback_spatial", "readback_channel", "main"],
    "controls": ["hsm_highpass", "direct_transfer", "bridge_full", "all_full", "no_core_compensation", "no_bridge_local"],
    "final": ["baseline", "main"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--sources", nargs="+", default=["isic2018"], choices=["isic2017", "isic2018"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epoch", "--epochs", dest="epochs", type=int,
                        help="Total epochs for every run; default comes from config")
    parser.add_argument("--stage", choices=list(STAGES), default="core")
    parser.add_argument("--variants", nargs="+", choices=["baseline", "main", *ABLATIONS],
                        help="Explicit variants override --stage")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Final test evaluation; omit during validation-only screening")
    parser.add_argument("--include-other-isic", action="store_true", help="Supplementary other-ISIC test")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epoch must be positive")
    for source in args.sources:
        for variant in args.variants or STAGES[args.stage]:
            config = ROOT / "configs" / f"main_{source}.yaml"
            for seed in args.seeds:
                run_dir = Path(args.output_root).resolve() / source / variant / f"seed_{seed}"
                command = [sys.executable, "-m", "skinmamba", "run" if args.evaluate else "train",
                    "--config", str(config), "--data-root", str(Path(args.data_root).resolve()),
                    "--run-dir", str(run_dir), "--device", args.device, "--seed", str(seed)]
                command.extend(["--" + variant] if variant in {"baseline", "main"} else ["--ablation", variant])
                if args.epochs is not None:
                    command.extend(["--epoch", str(args.epochs)])
                if args.include_other_isic:
                    command.append("--include-other-isic")
                for override in args.overrides:
                    command.extend(["--set", override])
                print(subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command), flush=True)
                if args.execute:
                    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
