"""Verify vendored author sources, or download exact pinned GitHub commits.

Default is offline verification. --download obtains source only, never weights,
and never runs upstream setup/install scripts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "EfficientViM": {
        "repository": "https://github.com/mlvlab/EfficientViM.git",
        "commit": "304340cb9c339b61669250d058525c9cdadd5e93",
        "files": {"classification/models/EfficientViM.py": "EfficientViM.original.py",
                  "classification/models/utils.py": "utils.original.py", "LICENSE": "LICENSE"},
    },
    "UPLiFT": {
        "repository": "https://github.com/mwalmer-umd/UPLiFT.git",
        "commit": "e58d213d79c125d5cceaa7af0fefb6a94677bf55",
        "files": {"uplift/uplift.py": "uplift.original.py", "LICENSE": "LICENSE",
                  "licenses/LayerNorm_LICENSE.txt": "LayerNorm_LICENSE.txt"},
    },
}


def git(checkout, *args):
    return subprocess.check_output(["git", "-C", str(checkout), *args])


def download(checkout_root):
    entries = []
    for name, spec in SOURCES.items():
        checkout = Path(checkout_root) / name
        if not (checkout / ".git").is_dir():
            checkout.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", str(checkout)], check=True)
            subprocess.run(["git", "-C", str(checkout), "fetch", "--depth", "1",
                            spec["repository"], spec["commit"]], check=True)
        git(checkout, "cat-file", "-e", spec["commit"] + "^{commit}")
        folder = ROOT / "third_party" / name
        folder.mkdir(parents=True, exist_ok=True)
        for original, filename in spec["files"].items():
            raw = git(checkout, "show", spec["commit"] + ":" + original)
            destination = folder / filename
            destination.write_bytes(raw)
            entries.append({"project": name, "repository": spec["repository"],
                            "commit": spec["commit"], "upstream_path": original,
                            "vendored_path": destination.relative_to(ROOT).as_posix(),
                            "sha256": hashlib.sha256(raw).hexdigest()})
    manifest = {"purpose": "Unmodified author source snapshots; runtime adaptations are in skinmamba/models/reconstruction.py",
                "files": entries}
    (ROOT / "third_party" / "RECONSTRUCTION_SOURCES.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def verify():
    path = ROOT / "third_party" / "RECONSTRUCTION_SOURCES.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        raw = (ROOT / entry["vendored_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Source snapshot changed: {entry['vendored_path']}")
    print(f"Verified {len(manifest['files'])} pinned author source/license files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--checkout-root", type=Path)
    args = parser.parse_args()
    if args.download:
        if args.checkout_root:
            download(args.checkout_root)
        else:
            with tempfile.TemporaryDirectory(prefix="skinmamba_upstream_") as directory:
                download(directory)
    verify()


if __name__ == "__main__":
    main()
