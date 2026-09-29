"""Pinned public inputs and explicit source-label handling."""

import csv
import hashlib
import json
from pathlib import Path
import urllib.request

from .model import FEATURES, feature_vector

COMMIT = "5ece2c65ba184faccf6c8cdccdc03132427c464b"
DEFAULT_DATA = Path("data/upstream/ufnet") / COMMIT


def source_manifest():
    return json.loads(Path(__file__).with_name("sources.json").read_text())


def verify_sources(root):
    result = {}
    for source in source_manifest():
        path = Path(root) / source["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {source['path']}")
        result[source["path"]] = actual
    return result


def fetch_sources(root):
    root = Path(root)
    for source in source_manifest():
        path = root / source["path"]
        content = path.read_bytes() if path.exists() else urllib.request.urlopen(source["url"], timeout=60).read()
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {source['path']}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    return verify_sources(root)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV needs unique column names")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no observations")
    return rows


def load_datasets(root):
    root = Path(root)
    hashes = verify_sources(root)
    sets = {name: set((root / f"data/{filename}_set_participants.txt").read_text().split())
            for name, filename in (("validation", "dev"), ("test", "test"), ("calibration", "calib"))}
    for left in sets:
        for right in sets:
            if left != right and sets[left] & sets[right]:
                raise ValueError(f"Participant split overlap: {left}/{right}")
    groups = {key: [] for key in ("train", "validation", "calibration", "test", "youtube")}
    excluded = []
    filenames = set()
    for row_number, row in enumerate(read_csv(root / "data/facial_expression_smile/facial_dataset.csv"), 2):
        pid, filename = row["ID"].strip(), row["Filename"].strip()
        if not pid or not filename or filename in filenames:
            raise ValueError("Missing participant/file identity or duplicate recording")
        filenames.add(filename)
        # The source Diagnosis column assigns positives to missing pd labels;
        # use only explicit yes/no labels for this experiment.
        raw_label = row["pd"].strip()
        if raw_label not in ("yes", "no"):
            excluded.append({"row": row_number, "id": pid, "label": raw_label,
                             "reason": "missing_or_ambiguous_pd_label"})
            continue
        split = next((key for key, ids in sets.items() if pid in ids), "train")
        groups[split].append({"id": pid, "recording": filename, "source_row": row_number,
                              "label": int(raw_label == "yes"), "features": feature_vector(row)})
    for row_number, row in enumerate(read_csv(root / "data/facial_expression_smile/youtube_PD_features_updated.csv"), 2):
        if row["pd"] not in ("y", "n"):
            raise ValueError("Unexpected YouTubePD label")
        groups["youtube"].append({"id": row["ID"], "recording": row["file_name"],
                                  "source_row": row_number, "label": int(row["pd"] == "y"),
                                  "features": feature_vector(row)})
    for split, records in groups.items():
        if {r["label"] for r in records} != {0, 1}:
            raise ValueError(f"Both classes required in {split}")
    audit = {"source_commit": COMMIT, "source_sha256": hashes, "features": list(FEATURES),
             "label_policy": {"no": 0, "yes": 1, "n": 0, "y": 1}, "excluded": excluded,
             "splits": {name: {"recordings": len(rows), "unique_ids": len({r['id'] for r in rows}),
                                "positive": sum(r["label"] for r in rows),
                                "negative": sum(1-r["label"] for r in rows)} for name, rows in groups.items()},
             "youtube_unit": "clip; person identities and cross-dataset identity overlap unverified"}
    return groups, audit
