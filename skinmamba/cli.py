from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from .config import ABLATIONS, apply_overrides, load_config, select_experiment, validate_config
from .utils import write_json, write_csv


def aggregate_runs(root, output):
    groups = {}
    seen = set()
    for path in sorted(Path(root).rglob("results.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if not {"config", "results", "checkpoint_sha256"}.issubset(report):
            continue
        # Keep threshold, data contents and hardware timing protocols separate.
        protocol = {k: report["config"][k] for k in ("data", "training", "loss")}
        protocol["data"] = {k:v for k,v in protocol["data"].items() if k != "root"}
        protocol["threshold"] = report["config"]["evaluation"]["threshold"]
        protocol["data_fingerprint"] = report["audit"]["fingerprint"]
        profile = report.get("profile", {})
        protocol["profiling"] = {key: profile.get(key) for key in (
            "device_name", "precision", "input_shape", "backend", "timing_scope", "torch_version",
            "warmup", "iterations", "flops_scope", "cpu_threads", "cuda_matmul_allow_tf32", "cudnn_allow_tf32")}
        protocol_key = json.dumps(protocol, sort_keys=True)
        identity = (report["checkpoint_sha256"], protocol_key)
        # Repeated evaluation under the same protocol is not an independent seed.
        if identity in seen:
            continue
        seen.add(identity)
        for row in report["results"]:
            # Older reports may contain clean subsets; summaries now use full tests only.
            if row.get("subset", "full") != "full":
                continue
            model_key = json.dumps(report["config"]["model"], sort_keys=True)
            key = (row["source"], row["target"], "full", model_key, protocol_key)
            groups.setdefault(key, []).append(row)
    result = []
    for key, rows in groups.items():
        if len({r["seed"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate seed runs in group {key[:3]}; select one run per seed")
        out = dict(source=key[0], target=key[1], subset=key[2], model=key[3], protocol=key[4],
                   seeds=",".join(str(r["seed"]) for r in rows), n_runs=len(rows))
        out["hd95_failed_count_sum"] = sum(r.get("hd95_failed_count", 0) for r in rows)
        out["hd95_finite_count_sum"] = sum(r.get("hd95_finite_count", 0) for r in rows)
        out["empty_prediction_count_sum"] = sum(r.get("empty_prediction_count", 0) for r in rows)
        out["hd95_definition"] = "per-seed finite-case mean; failures separately counted"
        out["flops_definition"] = "core arithmetic estimate, 2 FLOPs/MAC, includes selective scan; see profile.json"
        for metric in ("params", "flops", "size_mb", "fps", "dice", "iou", "miou", "accuracy", "sensitivity", "specificity", "f1", "hd95"):
            values = [float(r[metric]) for r in rows if r.get(metric) is not None]
            finite = [v for v in values if math.isfinite(v)]
            out[f"{metric}_mean"] = statistics.mean(values) if values else None
            out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 and len(finite) == len(values) else None
        result.append(out)
    if not result:
        raise ValueError(f"No evaluation results.json found below {root}")
    write_csv(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Skin Mamba reproducible experiment commands")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "run"):
        p = sub.add_parser(name, help="run=train then evaluate best on all 3 test domains")
        p.add_argument("--config", default="configs/baseline_isic2018.yaml")
        choice = p.add_mutually_exclusive_group()
        choice.add_argument("--baseline", dest="experiment", action="store_const", const="baseline",
                            help="Select the original baseline variant")
        choice.add_argument("--main", "--main-experiment", dest="experiment", action="store_const", const="main",
                            help="Select the main experiment: coverage-constrained lesion-adaptive sampling")
        choice.add_argument("--ablation", choices=ABLATIONS, help="Select one named ablation")
        p.add_argument("--epoch", "--epochs", dest="epochs", type=positive_int,
                       help="Total training epochs (not extra epochs when resuming)")
        p.add_argument("--seed", type=seed_value, help="Random seed, overriding YAML and --set seed")
        p.add_argument("--set", action="append", default=[], dest="overrides")
        p.add_argument("--data-root")
        p.add_argument("--run-dir", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--resume")
        p.add_argument("--stop-after-epoch", type=int)
        p.add_argument("--save-predictions", action="store_true")
        p.add_argument("--include-other-isic", action="store_true",
                       help="Also report the other ISIC test set as supplementary transfer")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root")
    p.add_argument("--output-dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", choices=["cuda", "reference"])
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--skip-profile", action="store_true")
    p.add_argument("--include-other-isic", action="store_true")
    p = sub.add_parser("audit")
    p.add_argument("--data-root", default="../data")
    p.add_argument("--output-dir", default="reports/data_audit")
    p = sub.add_parser("aggregate")
    p.add_argument("--root", default="runs")
    p.add_argument("--output", default="runs/aggregate.csv")
    args = parser.parse_args(argv)
    if args.command in ("train", "run"):
        from .engine import train
        config = training_config(args)
        print(f"experiment={config['name']} variant={config['model']['variant']} "
              f"source={config['data']['source']} epochs={config['training']['epochs']} seed={config['seed']}", flush=True)
        if args.command == "run" and args.stop_after_epoch and args.stop_after_epoch < config["training"]["epochs"]:
            raise ValueError("Use train (not run) for an interrupted session; final test evaluation follows completed training.")
        best = train(config, args.run_dir, args.device, args.resume, args.stop_after_epoch)
        if args.command == "run":
            from .evaluation import evaluate_checkpoint
            evaluate_checkpoint(best, device=args.device, save_predictions=args.save_predictions,
                                include_other_isic=args.include_other_isic)
    elif args.command == "evaluate":
        from .evaluation import evaluate_checkpoint
        evaluate_checkpoint(args.checkpoint, args.data_root, args.output_dir, args.device, args.backend,
                            args.save_predictions, not args.skip_profile,
                            include_other_isic=args.include_other_isic)
    elif args.command == "audit":
        from .data import make_manifest, audit_manifest
        manifest = make_manifest(args.data_root)
        write_json(Path(args.output_dir) / "manifest.json", manifest)
        for source in ("isic2017", "isic2018"):
            report = audit_manifest(manifest, source)
            write_json(Path(args.output_dir) / f"{source}.json", report)
            print(source, {k: v["total"] for k, v in report["tests"].items()})
    else:
        aggregate_runs(args.root, args.output)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("epochs must be a positive integer")
    return number


def seed_value(value):
    number = int(value)
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError("seed must be in [0, 2**32)")
    return number


def training_config(args):
    config = select_experiment(load_config(args.config), args.experiment, args.ablation)
    config = apply_overrides(config, args.overrides)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.seed is not None:
        config["seed"] = args.seed
    if args.data_root:
        config["data"]["root"] = args.data_root
    validate_config(config)
    return config


if __name__ == "__main__":
    main()
