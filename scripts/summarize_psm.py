"""Summarize source validation and frozen diagnostics without mixing domains."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skinmamba.config import load_config
from skinmamba.utils import write_csv


def summarize(root, output):
    output = Path(output)
    rows = []
    for status_path in sorted(Path(root).rglob("training_status.json")):
        run = status_path.parent
        c = load_config(run / "config.yaml")
        if c.get("experiment_suite") != "persistent_v1":
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        with (run / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
            history = list(csv.DictReader(handle))
        # Report the actual selected checkpoint, not the maximum of each metric.
        selected = next(r for r in history if int(r["epoch"]) == status["best_epoch"])
        rows.append(dict(source=c["data"]["source"], variant=c["name"], seed=c["seed"],
            complete=status["complete"], planned_epochs=status["planned_epochs"],
            completed_epochs=status["completed_epochs"], selected_epoch=status["best_epoch"],
            selection=status["selection"], val_loss=float(selected["val_loss"]),
            val_dice=float(selected["val_dice"]), train_seconds=sum(float(r["seconds"]) for r in history),
            run=str(run.resolve())))
    write_csv(output / "validation.csv", rows)
    changes, health = [], []
    for path in sorted(Path(root).rglob("state_changes.csv")):
        protocol = json.loads((path.parent / "protocol.json").read_text(encoding="utf-8"))
        main = protocol["checkpoints"]["main"]
        control = protocol["checkpoints"]["control"]
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                changes.append({**row, "source": protocol["source"], "phase": protocol["phase"],
                    "main": main["config"]["name"], "control": control["config"]["name"],
                    "checkpoint_sha256": main["sha256"], "control_sha256": control["sha256"],
                    "analysis_dir": str(path.parent.resolve())})
        with (path.parent / "state_health.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                health.append({**row, "source": protocol["source"], "phase": protocol["phase"],
                               "seed": main["config"]["seed"], "analysis_dir": str(path.parent.resolve())})
    write_csv(output / "state_changes_all.csv", changes)
    write_csv(output / "state_health_all.csv", health)
    print(f"Source validation: {len(rows)} runs. Paired state changes: {len(changes)} rows. Output: {output}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="reports/psm_summary")
    args = parser.parse_args()
    summarize(args.root, args.output)
