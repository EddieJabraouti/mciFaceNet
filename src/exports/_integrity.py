"""Verify each frozen checkpoint before loading it."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parent


def checkpoint(name):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    item = manifest["modules"][name]
    path = ROOT / item["checkpoint"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["checkpoint_sha256"]:
        raise ValueError(f"Frozen {name} checkpoint checksum mismatch")
    return path
