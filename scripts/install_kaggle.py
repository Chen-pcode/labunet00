"""Install official Mamba packages without replacing Kaggle's existing torch.

Sources: https://github.com/state-spaces/mamba
         https://github.com/Dao-AILab/causal-conv1d
This script never constructs a wheel URL or asserts that an untested version
pair is compatible. Official installers may download wheels or build locally.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment():
    import torch

    nvcc = shutil.which("nvcc")
    return {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvcc_path": nvcc,
        "nvcc_version": subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip() if nvcc else None,
        "mamba_ssm": version("mamba-ssm"), "causal_conv1d": version("causal-conv1d"),
    }


def package_spec(name, selected):
    if selected is None:
        return name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]*", selected):
        raise ValueError(f"Invalid {name} version: {selected!r}")
    return f"{name}=={selected}"


def run(command):
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mamba-version", help="Explicit version after verifying a compatible environment; default pip resolution.")
    parser.add_argument("--causal-conv1d-version", help="Explicit version; default pip resolution.")
    parser.add_argument("--requirements", type=Path, default=ROOT / "requirements.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "kaggle_install.json")
    args = parser.parse_args()
    report = {"status": "failed", "commands": [], "version_policy": "explicit CLI pins or pip-resolved official PyPI packages"}
    try:
        before = environment()
        report["before"] = before
        print(json.dumps(before, indent=2), flush=True)
        if not before["cuda_available"] or before["torch_cuda"] is None:
            raise RuntimeError("Enable a Kaggle NVIDIA GPU accelerator before installing. Existing torch must be a CUDA build.")
        if not args.requirements.is_file():
            raise FileNotFoundError(args.requirements)
        lines = args.requirements.read_text(encoding="utf-8").splitlines()
        # Reject direct torch requests; constraints below preserve the installed
        # torch stack while still installing missing transitive dependencies.
        for line in lines:
            candidate = line.split("#", 1)[0].strip()
            if not candidate:
                continue
            if candidate.startswith("-") or "://" in candidate or "/" in candidate or "\\" in candidate:
                raise ValueError("Use explicit PyPI requirements only; recursive/options/direct URL requirements are not supported by this torch-preserving installer.")
            name = re.split(r"[<>=!~;@\[\s]", candidate, maxsplit=1)[0].lower().replace("_", "-")
            if name in {"torch", "torchvision", "torchaudio", "mamba-ssm", "causal-conv1d"}:
                raise ValueError(f"Keep {name} out of requirements.txt; existing torch is preserved and scan packages are installed separately.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        constraints = args.output.parent / "kaggle_torch_constraints.txt"
        protected = {name: version(name) for name in ("torch", "torchvision", "torchaudio", "triton")}
        constraints.write_text("\n".join(f"{name}=={value}" for name,value in protected.items() if value) + "\n", encoding="utf-8")
        report["protected_packages"] = protected
        pip = [sys.executable, "-m", "pip", "install", "--constraint", str(constraints)]
        commands = [pip + ["-r", str(args.requirements)],
                    pip + ["packaging", "ninja", "einops", "setuptools", "wheel"],
                    pip + ["--no-build-isolation", package_spec("causal-conv1d", args.causal_conv1d_version)],
                    pip + ["--no-build-isolation", package_spec("mamba-ssm", args.mamba_version)]]
        if not before["nvcc_path"]:
            print("nvcc is unavailable. Installation requires an official compatible prebuilt artifact; source compilation cannot succeed without a CUDA toolkit.", flush=True)
        for command in commands:
            report["commands"].append(command)
            run(command)
        report["after"] = environment()
        for name, previous in protected.items():
            if previous is not None and version(name) != previous:
                raise RuntimeError(f"The protected {name} version changed unexpectedly.")
        # Use a fresh interpreter so a newly installed binary package is probed.
        run([sys.executable, "-c", "import torch, causal_conv1d, mamba_ssm; from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('Official scan packages import successfully; GPU verification remains required.')"])
        report["status"] = "installed_not_numerically_verified"
        print("Next: python scripts/verify_cuda.py --precision all", flush=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Install report: {args.output}", flush=True)
    return 0 if report["status"] == "installed_not_numerically_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
