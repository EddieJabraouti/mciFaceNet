"""YouTubePD-only fitting of the two original facial classification candidates."""

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import read_csv
from .model import FEATURES, FacialClassifier, feature_vector
from .train import as_arrays, calibrate, fit_ufnet, make_artifact, probabilities, write_json

SPLITS = ("train", "validation", "calibration", "test")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def video_id(url):
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        identifier = parsed.path.lstrip("/")
    elif parsed.hostname in ("youtube.com", "www.youtube.com") and parsed.path == "/watch":
        identifier = parse_qs(parsed.query).get("v", [""])[0]
    else:
        raise ValueError("Unsupported source video URL")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", identifier):
        raise ValueError("Invalid YouTube video identifier")
    return identifier


def partition(rows, seed):
    """Freeze source-video groups; never infer that a video ID is a person ID."""
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    assignments, provenance = {}, {}
    for vid, group in sorted(by_video.items()):
        labels = {r["label"] for r in group}
        if len(labels) != 1:
            raise ValueError(f"Conflicting labels for source video {vid}")
        known = {r["source_split"] for r in group if r["source_split"]}
        if len(known) > 1 or not known.issubset({"train", "val", "test"}):
            raise ValueError(f"Conflicting or unknown published splits for source video {vid}")
        if known:
            source = known.pop()
            assignments[vid] = "validation" if source == "val" else source
            provenance[vid] = "published_source_split"
    extras = sorted(set(by_video) - set(assignments))
    if extras:
        if any(by_video[vid][0]["label"] != 0 for vid in extras):
            raise ValueError("Expected unassigned supplementary source videos to be negative")
        train_ids, held_ids = train_test_split(extras, test_size=0.4, random_state=seed)
        val_ids, test_ids = train_test_split(held_ids, test_size=0.5, random_state=seed + 1)
        for split, ids in (("train", train_ids), ("validation", val_ids), ("test", test_ids)):
            assignments.update(dict.fromkeys(ids, split))
            provenance.update(dict.fromkeys(ids, "supplementary_video_group_60_20_20"))
    candidates = sorted(vid for vid, split in assignments.items() if split == "train")
    _, calibration_ids = train_test_split(
        candidates, test_size=0.2, random_state=seed + 2,
        stratify=[by_video[vid][0]["label"] for vid in candidates],
    )
    for vid in calibration_ids:
        assignments[vid] = "calibration"
        provenance[vid] += ";reserved_from_training_20_percent_stratified"
    groups = {split: [] for split in SPLITS}
    for row in rows:
        split = assignments[row["video_id"]]
        groups[split].append({**row, "split": split, "split_provenance": provenance[row["video_id"]]})
    audit_isolation(groups)
    return groups


def audit_isolation(groups):
    for split, rows in groups.items():
        if {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"Both classes are required in {split}")
    for field in ("video_id", "clip_id", "video_sha256"):
        memberships = defaultdict(set)
        for split, rows in groups.items():
            for row in rows:
                memberships[row[field]].add(split)
        if any(len(splits) > 1 for splits in memberships.values()):
            raise ValueError(f"Cross-partition overlap in {field}")
    # Identical feature vectors from different uploads must not masquerade as
    # independent held-out examples. This is a duplicate check, not identity proof.
    fingerprints = defaultdict(set)
    for split, rows in groups.items():
        for row in rows:
            fingerprints[tuple(row["features"])].add(split)
    if any(len(splits) > 1 for splits in fingerprints.values()):
        raise ValueError("An identical feature vector appears in different partitions")


def load_windows(features_dir, manifest_path):
    features_dir, manifest_path = Path(features_dir), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads((features_dir / "summary.json").read_text())
    if summary["source_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Extraction source manifest checksum mismatch")
    if summary["feature_columns"] != list(FEATURES):
        raise ValueError("Extraction feature schema differs from the classifiers")
    source = {r["clip_id"]: r for r in manifest["records"] if r["status"] == "downloaded"}
    accepted, rejected, seen = [], [], set()
    for row in read_csv(features_dir / "window_features.csv"):
        clip = row["clip_id"]
        if clip not in source:
            raise ValueError(f"Unknown source clip: {clip}")
        record = source[clip]
        start, end = float(row["start_seconds"]), float(row["end_seconds"])
        key = (clip, start, end)
        if key in seen:
            raise ValueError("Duplicate window identity")
        seen.add(key)
        if not (0 <= start < end <= record["expected_duration_seconds"] + 1e-8):
            raise ValueError("Window lies outside its annotated source clip")
        if not np.isclose(end - start, summary["window_seconds"]):
            raise ValueError("Unexpected window duration")
        if (row["pd"] not in ("y", "n") or row["pd"] != record["pd"]
                or int(row["label"]) != int(record["pd"] == "y")
                or row["source_video_url"] != record["url"]
                or row["source_split"] != (record["split_raw"] or "")):
            raise ValueError(f"Source provenance mismatch: {clip}")
        if row["quality_status"] == "review_required":
            rejected.append({"clip_id": clip, "start_seconds": start, "end_seconds": end,
                             "reason": row["quality_reasons"]})
            continue
        if row["quality_status"] != "passes_automatic_checks":
            raise ValueError("Unrecognized feature quality status")
        accepted.append({"clip_id": clip, "video_id": video_id(record["url"]),
                         "video_sha256": record["verification"]["sha256"],
                         "source_split": row["source_split"], "source_url": record["url"],
                         "start_seconds": start, "end_seconds": end,
                         "source_start_seconds": record["start_seconds"] + start,
                         "label": int(row["label"]), "features": feature_vector(row)})
    if len(seen) != summary["window_summaries"] or len(accepted) != summary["window_summaries_passing_checks"]:
        raise ValueError("Feature table counts disagree with extraction summary")
    return accepted, rejected, manifest, summary


def score(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    predicted = scores >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"n": len(labels), "positive": int(labels.sum()), "negative": int((labels == 0).sum()),
            "accuracy": float(accuracy_score(labels, predicted)), "precision": float(precision),
            "recall": float(recall), "f1": float(f1), "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]], "threshold": 0.5}


def averaged_score(rows, scores, field):
    buckets = defaultdict(list)
    labels = {}
    for row, probability in zip(rows, scores, strict=True):
        identifier = row[field]
        if identifier in labels and labels[identifier] != row["label"]:
            raise ValueError(f"Conflicting labels within {field}")
        labels[identifier] = row["label"]
        buckets[identifier].append(float(probability))
    ids = sorted(buckets)
    result = score([labels[i] for i in ids], [np.mean(buckets[i]) for i in ids])
    result["aggregation"] = f"Arithmetic mean of passing-window probabilities per {field}; not participant-level"
    return result


def write_csv(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_youtube(features_dir, manifest_path, recipe_path, output):
    output, features_dir = Path(output), Path(features_dir)
    recipe = json.loads(Path(recipe_path).read_text())
    cfg = recipe["ufnet_configuration"]
    if (cfg["model"], cfg["scaling_method"], cfg["optimizer"], cfg["num_epochs"]) != (
            "ShallowANN", "StandardScaler", "SGD", 64):
        raise ValueError("Expected the original facial-only training recipe")
    rows, rejected, manifest, extraction = load_windows(features_dir, manifest_path)
    groups = partition(rows, cfg["seed"])
    if min(Counter(r["label"] for r in groups["train"]).values()) < 6:
        raise ValueError("The original SMOTE recipe needs at least six real training windows per class")
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    counts = {split: {"windows": len(group), "clips": len({r["clip_id"] for r in group}),
                      "source_videos": len({r["video_id"] for r in group}),
                      "positive_windows": sum(r["label"] for r in group),
                      "negative_windows": sum(1 - r["label"] for r in group)}
              for split, group in groups.items()}
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "YouTubePD only",
        "input_unit": f"{extraction['window_seconds']:g}-second facial window, 42 summary features",
        "quality_rule": "Only passes_automatic_checks windows; rules frozen before classification",
        "partition_policy": "Preserve published train/val/test video groups; assign supplementary negative videos "
                            "60/20/20 using seeds seed and seed+1; reserve 20% of training video groups "
                            "for calibration, stratified by label, seed+2.",
        "seed": cfg["seed"], "splits": counts,
        "primary_selection": "Lowest raw window-level validation log loss; validation selects neural epoch",
        "calibration": "Platt scaling on separate calibration video groups; fixed threshold 0.5",
        "ufnet_configuration": cfg, "logistic_configuration": recipe["logistic_reference"],
        "training_data": "Only current YouTubePD training windows; no UFNet samples or weights",
        "source_commit": manifest["upstream_commit"],
        "input_sha256": {str(p): sha256(p) for p in (
            Path(manifest_path), Path(recipe_path), features_dir / "window_features.csv",
            features_dir / "protocol.json", features_dir / "summary.json")},
        "source_code_sha256": {p.name: sha256(p) for p in (
            Path(__file__), Path(__file__).with_name("train.py"), Path(__file__).with_name("model.py"))},
        "versions": {name: version(name) for name in ("numpy", "scikit-learn", "imbalanced-learn", "torch")},
        "limitations": ["Video IDs are not participant IDs; cross-upload identity overlap remains unverified.",
                        "Source PD labels are the target; performance does not establish MCI/cognitive-decline detection.",
                        "Features are an unverified numerical reimplementation of UFNet extraction.",
                        "OpenFace centered smoothing uses neighboring frames; every source video remains in one split.",
                        "Windows within clips/videos are correlated; report clip/video aggregation as well.",
                        "Native frame rates differ and the entropy features depend on valid/active frame counts.",
                        "Supplementary negatives change class proportions; accuracy alone can be misleading.",
                        "Single fixed split with few positive test videos; exploratory performance only."],
    }
    # Persist assignments and the fixed recipe before any model sees outcomes.
    write_json(output / "protocol.json", protocol)
    write_json(output / "exclusions.json", rejected)
    assignments = [{k: v for k, v in row.items() if k != "features"}
                   for split in SPLITS for row in groups[split]]
    write_csv(output / "splits.csv", assignments)
    x_train, y_train = as_arrays(groups["train"])
    scaler = StandardScaler().fit(x_train)
    x_scaled = scaler.transform(x_train)
    neural, history = fit_ufnet(x_scaled, y_train, groups["validation"], scaler, cfg)
    logistic_cfg = recipe["logistic_reference"]
    logistic = LogisticRegression(C=logistic_cfg["C"], solver=logistic_cfg["solver"],
                                  max_iter=logistic_cfg["max_iter"])
    logistic.fit(x_scaled, y_train)
    reference = FacialClassifier(make_artifact("logistic_reference", scaler, logistic.coef_, logistic.intercept_[0], 0))
    candidates = {"logistic_reference": reference, "ufnet_shallow_dropout": neural}
    for model in candidates.values():
        model.artifact.update(source_commit=manifest["upstream_commit"], input_unit=protocol["input_unit"],
                              dataset="YouTubePD only", extractor_protocol_sha256=sha256(features_dir / "protocol.json"))
    # Reconstruct to keep fingerprints consistent with the updated provenance.
    candidates = {name: FacialClassifier(model.artifact) for name, model in candidates.items()}
    selection = {name: score([r["label"] for r in groups["validation"]], probabilities(model, groups["validation"]))
                 for name, model in candidates.items()}
    selected = min(selection, key=lambda name: selection[name]["log_loss"])
    write_json(output / "selection.json", {"selected": selected, "raw_validation": selection})
    write_json(output / "training_history.json", history)
    calibrated = {name: calibrate(model, groups["calibration"]) for name, model in candidates.items()}
    reports, predictions, comparisons, export_checks = {}, [], [], {}
    for name in candidates:
        reports[name] = {}
        for stage, model in (("raw", candidates[name]), ("calibrated", calibrated[name])):
            path = output / "models" / f"{name}_{stage}.json"
            model.export(path)
            reloaded = FacialClassifier.load(path)
            expected = probabilities(model, rows)
            difference = float(np.max(np.abs(expected - probabilities(reloaded, rows))))
            if difference != 0:
                raise AssertionError("Export/reload changed predictions")
            export_checks[f"{name}_{stage}"] = {"max_absolute_difference": difference, "sha256": sha256(path)}
            for split, group in groups.items():
                scores = probabilities(model, group)
                metrics = {"window": score([r["label"] for r in group], scores),
                           "clip": averaged_score(group, scores, "clip_id"),
                           "source_video": averaged_score(group, scores, "video_id")}
                reports[name].setdefault(split, {})[stage] = metrics
                for unit, values in metrics.items():
                    comparisons.append({"model": name, "stage": stage, "split": split, "unit": unit,
                                        **{k: values[k] for k in ("n", "positive", "negative", "accuracy", "precision", "recall", "f1", "log_loss")}})
                for row, probability in zip(group, scores, strict=True):
                    predictions.append({"model": name, "stage": stage, "split": split,
                                        "clip_id": row["clip_id"], "video_id": row["video_id"],
                                        "start_seconds": row["start_seconds"], "end_seconds": row["end_seconds"],
                                        "label": row["label"], "p_impaired": float(probability),
                                        "predicted": int(probability >= 0.5)})
    calibrated[selected].export(output / "classifier.json")
    prior = float(y_train.mean())
    majority = {split: score([r["label"] for r in group], np.full(len(group), prior)) for split, group in groups.items()}
    report = {"selected_model": selected, "models": reports, "constant_training_prior": majority,
              "export_verification": export_checks, "splits": counts,
              "interpretation": {"train": "fitting performance on real windows, not SMOTE samples",
                                 "validation": "model/epoch selection", "calibration": "calibrator fitting",
                                 "test": "source-video-disjoint test; participant independence unverified"}}
    write_json(output / "metrics.json", report)
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "comparison.csv", comparisons)
    return {"selected_model": selected, "splits": counts,
            "calibrated_test_window": {name: reports[name]["test"]["calibrated"]["window"] for name in reports},
            "output": str(output)}
