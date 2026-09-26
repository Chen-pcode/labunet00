"""Fixed-folder splits, paired transforms, and exact-duplicate leakage audit.

IDs and decoded RGB hashes detect exact duplication, not lesions/patients with
different photographs. No raw files are moved, removed or repartitioned.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch.utils.data import Dataset, DataLoader

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS = {"isic2017": ("train", "val", "test"), "isic2018": ("train", "val", "test"), "ph2": ("test",)}


def sample_id(path):
    name = Path(path).stem.lower()
    return name.removesuffix("_segmentation").removesuffix("_mask")


def paired_files(root, domain, split):
    folder = Path(root) / domain / split
    maps = []
    for kind in ("images", "masks"):
        directory = folder / kind
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        mapping = {}
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in EXTENSIONS:
                key = sample_id(path)
                if key in mapping:
                    raise ValueError(f"Duplicate sample ID {key} in {directory}")
                mapping[key] = path
        maps.append(mapping)
    images, masks = maps
    if not images or images.keys() != masks.keys():
        raise ValueError(f"Unpaired or empty split {folder}: missing masks={sorted(images.keys()-masks.keys())[:10]}, missing images={sorted(masks.keys()-images.keys())[:10]}")
    return [(key, images[key], masks[key]) for key in sorted(images)]


def decoded_hash(path, mode):
    with Image.open(path) as image:
        arr = np.asarray(image.convert(mode))
    digest = hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest()
    return digest, arr


def make_manifest(root):
    root = Path(root).resolve()
    records = []
    for domain, splits in SPLITS.items():
        for split in splits:
            for key, image, mask in paired_files(root, domain, split):
                image_hash, image_array = decoded_hash(image, "RGB")
                mask_hash, mask_array = decoded_hash(mask, "L")
                if image_array.shape[:2] != mask_array.shape:
                    raise ValueError(f"Image/mask size mismatch: {image} vs {mask}")
                records.append(dict(domain=domain, split=split, id=key,
                    image=image.relative_to(root).as_posix(), mask=mask.relative_to(root).as_posix(),
                    image_hash=image_hash, mask_hash=mask_hash,
                    height=mask_array.shape[0], width=mask_array.shape[1],
                    mask_values=np.unique(mask_array).tolist(), empty_mask=not bool(mask_array.any())))
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {"schema_version": 1, "root_at_audit": str(root), "fingerprint": hashlib.sha256(canonical.encode()).hexdigest(), "records": records}


def duplicate_matches(exposed, target):
    ids, hashes = {r["id"] for r in exposed}, {r["image_hash"] for r in exposed}
    return [r["id"] for r in target if r["id"] in ids or r["image_hash"] in hashes]


def audit_manifest(manifest, source):
    records = manifest["records"]
    select = lambda domain, split: [r for r in records if r["domain"] == domain and r["split"] == split]
    train, val = select(source, "train"), select(source, "val")
    report = {"source": source, "fingerprint": manifest["fingerprint"],
              "source_train_val_overlap": duplicate_matches(train, val), "tests": {}}
    for domain in SPLITS:
        test = select(domain, "test")
        duplicates = duplicate_matches(train + val, test)
        report["tests"][domain] = {"total": len(test), "overlap_ids": duplicates,
            "clean_count": len(test) - len(duplicates), "clean_external": domain != source and not duplicates}
    report["limitation"] = "IDs and decoded-image exact hashes only; no patient/lesion metadata or near-duplicate guarantee."
    return report


def validate_source_split(report, policy="exclude"):
    if policy not in ("exclude", "error"):
        raise ValueError("validation_overlap_policy must be exclude or error")
    if report["source_train_val_overlap"] and policy == "error":
        raise ValueError("Source train/val leakage detected; see audit report")


class SkinDataset(Dataset):
    def __init__(self, root, records, image_size=256, normalization="official_minmax255",
                 augmentation="none", seed=42):
        self.root, self.records = Path(root), list(records)
        self.image_size, self.normalization = image_size, normalization
        self.augmentation, self.seed, self.epoch = augmentation, seed, 0
        if normalization not in ("official_minmax255", "unit"):
            raise ValueError(f"Unknown normalization: {normalization}")
        if augmentation not in ("none", "official"):
            raise ValueError(f"Unknown augmentation: {augmentation}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        size = (self.image_size, self.image_size)
        with Image.open(self.root / record["image"]) as im:
            image = np.asarray(im.convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
        with Image.open(self.root / record["mask"]) as im:
            mask = np.asarray(im.convert("L").resize(size, Image.Resampling.NEAREST))
        # Support both {0,1} and {0,255}; grayscale boundaries are explicitly binarized.
        mask = (mask >= (1 if mask.max() <= 1 else 128)).astype(np.float32)
        if self.normalization == "official_minmax255":
            lo, hi = float(image.min()), float(image.max())
            image = (image - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(image)
        else:
            image /= 255
        if self.augmentation == "official":
            # Local RNG makes epoch-boundary resume independent of DataLoader workers.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            if rng.random() > .5:
                k, axis = int(rng.integers(0, 4)), int(rng.integers(0, 2))
                image, mask = np.flip(np.rot90(image, k), axis), np.flip(np.rot90(mask, k), axis)
            if rng.random() > .5:
                angle = int(rng.integers(20, 80))
                image = ndimage.rotate(image, angle, order=0, reshape=False)
                mask = ndimage.rotate(mask, angle, order=0, reshape=False)
        return {"image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
                "mask": torch.from_numpy(np.ascontiguousarray(mask[None])), "id": record["id"]}


def make_loader(dataset, batch_size, workers=0, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed + dataset.epoch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False, generator=generator)
