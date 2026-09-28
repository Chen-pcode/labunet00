"""Versioned PSM experiment matrix. Dry run by default; screening never tests PH2."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skinmamba.persistent_experiments import PSM_PRESETS

STAGES = {
    "screen": ["psm_baseline", "psm_baseline_aug", "psm_memory_aug", "psm_main"],
    "ablation": ["psm_independent_aug", "psm_independent_dom", "psm_memory", "psm_feature_dom",
                 "psm_moments_only", "psm_no_variance"],
    "formal": list(PSM_PRESETS),
}


def run_streaming(command):
    """Relay child output through this process so notebook cells see it live."""
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"}) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="screen")
    parser.add_argument("--variants", nargs="+", choices=PSM_PRESETS)
    parser.add_argument("--sources", nargs="+", choices=["isic2017", "isic2018"], default=["isic2018"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epoch", "--epochs", dest="epochs", type=int, default=100)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="runs/psm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Final fixed-protocol test evaluation only")
    args = parser.parse_args(argv)
    if args.epochs < 1 or any(not 0 <= s < 2**32 for s in args.seeds):
        parser.error("Positive epochs and seeds in [0,2**32) required")
    if args.stage == "screen" and args.evaluate:
        parser.error("Screen uses source validation only; --evaluate requires --stage ablation/formal")
    variants = args.variants or STAGES[args.stage]
    total = len(args.sources) * len(variants) * len(args.seeds)
    print(f"PSM matrix: {total} experiments, {args.epochs} epochs each. "
          + ("Executing training." if args.execute else "DRY RUN: commands only; add --execute to train."), flush=True)
    current = 0
    for source in args.sources:
        for variant in variants:
            for seed in args.seeds:
                current += 1
                print(f"\n[Experiment {current}/{total}] source={source} variant={variant} "
                      f"seed={seed} epochs={args.epochs}", flush=True)
                run = Path(args.output_root).resolve() / source / variant / f"seed_{seed}"
                command = [sys.executable, "-u", "-m", "skinmamba", "run" if args.evaluate else "train",
                           "--config", str(ROOT / "configs/psm" / f"{source}.yaml"),
                           "--ablation", variant, "--epoch", str(args.epochs), "--seed", str(seed),
                           "--data-root", str(Path(args.data_root).resolve()), "--run-dir", str(run),
                           "--device", args.device]
                for override in args.overrides:
                    command.extend(["--set", override])
                print(subprocess.list2cmdline(command) if sys.platform == "win32" else shlex.join(command), flush=True)
                if args.execute:
                    run_streaming(command)
                    print(f"[Experiment {current}/{total}] Complete: {run}", flush=True)


if __name__ == "__main__":
    main()
