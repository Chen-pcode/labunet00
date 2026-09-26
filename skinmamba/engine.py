from __future__ import annotations

import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml

from .data import SkinDataset, make_loader, make_manifest, audit_manifest, validate_source_split
from .losses import BCEDiceLoss
from .models import build_model
from .utils import (seed_everything, rng_state, restore_rng, write_json, write_csv,
                    atomic_torch_save, load_checkpoint, environment)


def get_device(requested):
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. CPU smoke tests require --set model.backend=reference --device cpu.")
    return device


def amp_context(device, precision):
    if precision == "amp_fp16" and device.type != "cuda":
        raise ValueError("amp_fp16 is supported only on CUDA")
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=precision == "amp_fp16")


def validate(model, loader, criterion, device, precision, threshold):
    model.eval()
    loss_sum, count, dice_sum = 0.0, 0, 0.0
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            with amp_context(device, precision):
                logits = model(image)
            loss = criterion(logits, target)
            pred, truth = logits.float().sigmoid() >= threshold, target >= .5
            tp = (pred & truth).sum((1, 2, 3)).float()
            denom = pred.sum((1, 2, 3)) + truth.sum((1, 2, 3))
            dice = torch.where(denom > 0, 2 * tp / denom.clamp_min(1), torch.ones_like(tp))
            n = image.shape[0]
            count += n
            loss_sum += loss.item() * n
            dice_sum += dice.sum().item()
    return {"val_loss": loss_sum / count, "val_dice": dice_sum / count}


def resume_signature(config):
    config = copy.deepcopy(config)
    config.pop("name", None)
    config["data"].pop("root", None)
    config["training"].pop("workers", None)
    config.pop("profiling", None)
    config["evaluation"] = {"threshold": config["evaluation"]["threshold"]}
    return config


def train(config, run_dir, device="cuda", resume=None, stop_after_epoch=None):
    """Train with fixed source train/val; checkpoint at complete epoch boundaries.

    stop_after_epoch is only a smoke/session-boundary control. It does not alter
    the planned schedule or total epochs and therefore allows exact CPU resume.
    """
    device, run_dir = get_device(device), Path(run_dir)
    if (run_dir / "last.pt").exists() and resume is None:
        raise FileExistsError(f"Run already exists: {run_dir}; use --resume or a new --run-dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"], config["training"].get("strict_determinism", False))
    manifest = make_manifest(config["data"]["root"])
    audit = audit_manifest(manifest, config["data"]["source"])
    write_json(run_dir / "data_audit.json", audit)
    validate_source_split(audit, config["data"].get("validation_overlap_policy", "exclude"))
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint:
        if checkpoint["manifest"]["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Dataset content/splits changed since checkpoint; resume refused")
        if resume_signature(checkpoint["config"]) != resume_signature(config):
            raise ValueError("Training configuration changed since checkpoint; resume refused")
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(run_dir / "environment.json", environment())
    records = manifest["records"]
    ds_args = dict(root=config["data"]["root"], image_size=config["data"]["image_size"],
                   normalization=config["data"]["normalization"], seed=config["seed"])
    source = config["data"]["source"]
    train_ds = SkinDataset(records=[r for r in records if r["domain"] == source and r["split"] == "train"],
                           augmentation=config["data"]["augmentation"], **ds_args)
    excluded_val = set(audit["source_train_val_overlap"])
    val_ds = SkinDataset(records=[r for r in records if r["domain"] == source and r["split"] == "val"
                                  and r["id"] not in excluded_val], **ds_args)
    if not len(val_ds):
        raise ValueError("No validation samples remain after overlap exclusion")
    write_json(run_dir / "validation_protocol.json", {"excluded_overlap_ids": sorted(excluded_val),
        "n_train": len(train_ds), "n_validation_used": len(val_ds), "raw_files_modified": False})
    options = config["training"]
    val_loader = make_loader(val_ds, options["batch_size"], options["workers"])
    model = build_model(config).to(device)
    criterion = BCEDiceLoss(**config.get("loss", {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"],
                                 betas=(.9, .999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options["t_max"], eta_min=options["eta_min"])
    scaler = torch.amp.GradScaler("cuda", enabled=options["precision"] == "amp_fp16")
    start, best_epoch, history = 0, -1, []
    best = float("inf") if options["selection"] == "val_loss" else -float("inf")
    best_weights = None
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best, best_epoch = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["best_epoch"]
        history = checkpoint["history"]
        best_weights = checkpoint["best_model"]
        restore_rng(checkpoint["rng"])
        # Carry the best validation weights even if resuming into a new directory.
        best_artifact = dict(model=best_weights, config=config, epoch=best_epoch,
                             manifest=manifest, best=best, best_epoch=best_epoch)
        atomic_torch_save(best_artifact, run_dir / "best.pt")
    limit = min(options["epochs"], stop_after_epoch or options["epochs"])
    for epoch in range(start, limit):
        train_ds.epoch = epoch
        loader = make_loader(train_ds, options["batch_size"], options["workers"], shuffle=True, seed=config["seed"])
        model.train()
        total_loss, seen, started = 0.0, 0, time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        for step, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            with amp_context(device, options["precision"]):
                logits = model(image)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}, step {step}")
            scaler.scale(loss).backward()
            if options.get("grad_clip", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), options["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(image)
            seen += len(image)
            if (step + 1) % options.get("log_interval", 20) == 0:
                print(f"epoch={epoch+1} batch={step+1}/{len(loader)} train_loss={total_loss/seen:.5f}", flush=True)
        scores = validate(model, val_loader, criterion, device, options["precision"], config["evaluation"]["threshold"])
        scheduler.step()
        score = scores[options["selection"]]
        improved = score < best if options["selection"] == "val_loss" else score > best
        if improved:
            best, best_epoch = score, epoch
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        row = dict(epoch=epoch + 1, train_loss=total_loss / seen, **scores, lr=lr,
                   seconds=time.perf_counter() - started, best_epoch=best_epoch + 1)
        history.append(row)
        state = dict(model=model.state_dict(), best_model=best_weights, optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=rng_state(),
                     epoch=epoch, best=best, best_epoch=best_epoch, config=config, manifest=manifest, history=history)
        atomic_torch_save(state, run_dir / "last.pt")
        if improved:
            atomic_torch_save(dict(model=best_weights, config=config, epoch=epoch, manifest=manifest,
                                   best=best, best_epoch=best_epoch), run_dir / "best.pt")
        write_csv(run_dir / "history.csv", history)
        print(json.dumps(row), flush=True)
    if best_weights is None:
        raise ValueError("No epoch was trained and no checkpoint was resumed")
    atomic_torch_save(best_weights, run_dir / "weights.pt")
    write_json(run_dir / "training_status.json", {"completed_epochs": len(history), "planned_epochs": options["epochs"],
        "complete": len(history) >= options["epochs"], "best_epoch": best_epoch + 1,
        "selection": options["selection"], "best_validation_value": best})
    return run_dir / "best.pt"
