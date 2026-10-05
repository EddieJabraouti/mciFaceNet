"""A prespecified eleven-token experiment using existing frozen representations."""

import copy
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_datasets, verify_sources
from .fusion_data import checked, face_arrays, load_typing, load_typing_sequences, sha256
from .fusion_fixed import audit_pairs
from .fusion_gallery import GalleryFusionClassifier, GalleryFusionNetwork, build_galleries
from .fusion_regularized import RegularizedFusionClassifier
from .fusion_regularized_train import ARCHITECTURE, CONFIG, fit_regularized
from .fusion_train import classification_metrics, participant_metrics, write_json
from .model import FEATURES


def assert_disjoint(galleries):
    for field in ("typing_id", "face_id", "typing_source_index"):
        sets = [set(d[field].ravel()) for d in galleries.values()]
        if sum(map(len, sets)) != len(set.union(*sets)):
            raise ValueError(f"Gallery partitions overlap: {field}")


def reference_scores(model, face, typing):
    """Identical gallery inputs; classify each typing vector, then average ten."""
    return model.predict_proba(np.repeat(face, 10, axis=0), typing.reshape(-1, 128))[:, 1].reshape(-1, 10).mean(1)


def train_gallery_fusion(data_dir, typing_run, previous_run, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = {**copy.deepcopy(CONFIG), "batch_size": 32}
    architecture = dict(ARCHITECTURE)
    previous_run, out = Path(previous_run), Path(output)
    freeze = json.loads((previous_run / "freeze.json").read_text())
    checked(previous_run / "protocol.json", freeze["protocol_sha256"])
    checked(previous_run / "selection.json", freeze["selection_sha256"])
    previous_protocol = json.loads((previous_run / "protocol.json").read_text())
    reference_path = previous_run / "models/regularized.pt"
    checked(reference_path, freeze["models"]["regularized"])
    reference = RegularizedFusionClassifier.load(reference_path)
    rows, face_audit = load_datasets(data_dir)
    typing, typing_audit = load_typing(typing_run)
    sequences, sequence_audit = load_typing_sequences(typing_run, typing)
    if (face_audit["source_sha256"] != previous_protocol["face_data"]["source_sha256"]
            or typing_audit["hashes"] != previous_protocol["typing_data"]["hashes"]):
        raise ValueError("Previous fusion uses different source data")
    raw, galleries, counts, excluded = {}, {}, {}, {}
    seeds = {"train": 2026100501, "validation": 2026100502, "test": 2026100503}
    for role, seed in seeds.items():
        face = face_arrays(rows[role])
        checked(previous_run / f"pairs_{role}.npz", freeze["pair_hashes"][role])
        with np.load(previous_run / f"pairs_{role}.npz", allow_pickle=False) as source:
            pairs = {k: source[k] for k in source.files}
        audit_pairs(face, typing[role], pairs["face_index"], pairs["typing_index"])
        for key, expected in {
            "typing_id": typing[role]["owner"][pairs["typing_index"]],
            "typing_source_index": typing[role]["source_index"][pairs["typing_index"]],
            "typing_cohort": typing[role]["cohort"][pairs["typing_index"]],
            "y": typing[role]["y"][pairs["typing_index"]],
            "face_id": face["owner"][pairs["face_index"]],
            "face_source_row": face["source_index"][pairs["face_index"]],
            "face_recording": face["recording"][pairs["face_index"]],
        }.items():
            if not np.array_equal(pairs[key], expected):
                raise ValueError(f"Source metadata mismatch: {role}/{key}")
        g, audit, dropped = build_galleries(pairs, seed)
        fi, ti = g["face_index"], g["typing_index"]
        if not np.all(typing[role]["owner"][ti] == g["typing_id"][:, None]):
            raise ValueError("Gallery mixes typing participants")
        if not np.all(typing[role]["y"][ti] == face["y"][fi, None]):
            raise ValueError("Gallery labels disagree")
        if len(np.unique(g["typing_source_index"])) != ti.size:
            raise ValueError("Gallery repeats a typing window")
        g["keystroke_lengths"] = sequences[role]["lengths"][ti]
        audit.update({"full_50_key_sequences": int((g["keystroke_lengths"] == 50).sum()),
                      "partial_sequences": int((g["keystroke_lengths"] < 50).sum()),
                      "keys_per_gallery_min": int(g["keystroke_lengths"].sum(1).min()),
                      "keys_per_gallery_median": float(np.median(g["keystroke_lengths"].sum(1))),
                      "keys_per_gallery_max": int(g["keystroke_lengths"].sum(1).max())})
        galleries[role], counts[role], excluded[role] = g, audit, dropped
        raw[role] = {"face": face["x"][fi], "typing": typing[role]["x"][ti],
                     "owner": g["typing_id"], "y": g["y"]}
    assert_disjoint(galleries)
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    limitations = [
        "Synthetic same-label partner assignments; facial and typing subjects are different people",
        "Participant partners and splits retained; one facial recording chosen per gallery cycles within that same facial participant",
        "Exactly ten unique cached embeddings per gallery; incomplete gallery tails discarded, no overlapping galleries",
        "Source sequences may contain fewer than 50 keys; galleries are not necessarily 500 keys or consecutive in time",
        "No cross-window chronology is asserted; galleries have no positional encodings",
        "Previously inspected test participants; this is exploratory reuse, not a new untouched test set",
        "Frozen two-token reference is rescored on identical gallery inputs but was trained on the larger original window pool",
        "Batch size is 32 galleries, compared with 256 windows in the earlier run; this is not a single-factor architecture comparison",
        "UFNet prompted-smiling features remain the facial source; new video-extractor numerical parity is unverified",
        "Uncalibrated probabilities; positive class follows source PD labels",
    ]
    source = Path(__file__).parent
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "experiment": "Joint eleven-token attention: one face plus ten same-typist embeddings",
                "settings": cfg, "architecture": architecture, "gallery_size": 10,
                "gallery_seeds": seeds, "counts": counts,
                "previous_run": str(previous_run.resolve()),
                "previous_protocol_sha256": sha256(previous_run / "protocol.json"),
                "previous_model_sha256": sha256(reference_path),
                "previous_pair_hashes": freeze["pair_hashes"],
                "typing_run": str(Path(typing_run).resolve()),
                "face_data": face_audit, "typing_data": typing_audit, "sequence_data": sequence_audit,
                "preprocessing": "Reuse unchanged training-only face and typing scalers from the two-token reference",
                "grouping": "Sort each typist's source indices, permute with fixed split seed, take disjoint tens; permute/cycle their existing facial recordings",
                "attention": "Full self-attention over 11 projected tokens, no positions; concatenate updated face with mean updated typing tokens AFTER attention",
                "selection": "Three fixed seeds; per-seed participant validation log loss, patience 8; equal ensemble, no candidate search",
                "training": "All retained galleries once per epoch; equal participant loss weighting; train-only dropout and label smoothing",
                "reporting": "Gallery predictions averaged equally within participant; threshold 0.5",
                "calibration": "Uncalibrated", "limitations": limitations,
                "source_code_sha256": {p.name: sha256(p) for p in source.glob("*.py")},
                "versions": {name: version(name) for name in ("numpy", "torch", "scikit-learn")}}
    write_json(out / "protocol.json", protocol)
    write_json(out / "scaling.json", reference.bundle["scaling"])
    (out / "source").mkdir()
    for path in source.glob("*.py"):
        (out / "source" / path.name).write_bytes(path.read_bytes())
    for role, g in galleries.items():
        np.savez_compressed(out / f"galleries_{role}.npz", **g, discarded_typing_source_index=excluded[role])
    torch.manual_seed(0)
    bundle = {"format_version": 3, "classes": ["control", "impaired"], "threshold": .5,
              "face_features": list(FEATURES), "typing_dimensions": 128, "gallery_size": 10,
              "typing_encoder_sha256": typing_audit["encoder_sha256"],
              "architecture": architecture, "scaling": reference.bundle["scaling"],
              "states": [GalleryFusionNetwork(**architecture).state_dict()], "seeds": cfg["seeds"],
              "input_unit": "face[N,42], typing[N,10,128]; exactly ten distinct same-person embeddings",
              "face_representation": "UFNet prompted-smiling 42-summary schema; custom extractor parity unverified",
              "pooling": "Full eleven-token attention then face token plus mean typing tokens; permutation invariant"}
    processor = GalleryFusionClassifier(bundle)
    prepared = {role: {"tensors": processor.prepare(raw[role]["face"], raw[role]["typing"]),
                       "y": raw[role]["y"], "owner": raw[role]["owner"]} for role in ("train", "validation")}
    states, histories = [], []
    for seed in cfg["seeds"]:
        print(f"Training eleven-token gallery attention, seed {seed}", flush=True)
        state, trace = fit_regularized(architecture, seed, prepared["train"], prepared["validation"], cfg,
                                       network_class=GalleryFusionNetwork)
        states.append(state)
        histories.append(trace)
        write_json(out / "training_history.json", histories)
        print(f"Selected epoch {trace['best_epoch']}; stopped at {len(trace['epochs'])}", flush=True)
    bundle["states"] = states
    model = GalleryFusionClassifier(bundle)
    model_path = out / "models/gallery.pt"
    digest = model.export(model_path)
    val = raw["validation"]
    p = model.predict_proba(val["face"], val["typing"])[:, 1]
    write_json(out / "selection.json", {"selected": "gallery", "sha256": digest,
               "epochs": [t["best_epoch"] for t in histories],
               "validation": participant_metrics(val["y"], p, val["owner"])})
    write_json(out / "freeze.json", {"protocol_sha256": sha256(out / "protocol.json"),
               "selection_sha256": sha256(out / "selection.json"), "models": {"gallery": digest},
               "gallery_hashes": {role: sha256(out / f"galleries_{role}.npz") for role in galleries}})
    # All states and selection decisions are frozen before the first test prediction.
    reloaded, reports, parity = GalleryFusionClassifier.load(model_path), {}, {}
    for role, data in raw.items():
        p = model.predict_proba(data["face"], data["typing"])[:, 1]
        again = reloaded.predict_proba(data["face"], data["typing"])[:, 1]
        parity[role] = float(np.max(np.abs(p-again)))
        if parity[role] != 0:
            raise AssertionError("Export/reload changed gallery predictions")
        predictions = {"gallery": p, "previous_two_token": reference_scores(reference, data["face"], data["typing"])}
        reports[role] = {name: {"gallery": classification_metrics(data["y"], score),
                                "participant": participant_metrics(data["y"], score, data["owner"])}
                         for name, score in predictions.items()}
        np.savez_compressed(out / f"predictions_{role}.npz", y=data["y"], typing_id=data["owner"], **predictions)
    np.savez_compressed(out / "example_input.npz", **{k: raw["test"][k][:5] for k in ("face", "typing")})
    result = {"experiment": protocol["experiment"], "reports": reports, "counts": counts,
              "parameters_per_network": histories[0]["parameters"], "ensemble_size": len(states),
              "export_bytes": model_path.stat().st_size, "export_max_absolute_difference": parity,
              "threshold": .5, "limitations": limitations}
    write_json(out / "metrics.json", result)
    with (out / "comparison.csv").open("w", newline="") as stream:
        fields = ["split", "unit", "model", "n", "accuracy", "precision", "recall", "f1"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for role, records in reports.items():
            for name, units in records.items():
                for unit, metrics in units.items():
                    writer.writerow({"split": role, "unit": unit, "model": name, **{k: metrics[k] for k in fields[3:]}})
    # A modular compatibility contract, not an export of the camera pipeline.
    write_json(out / "pipeline_manifest.json", {
        "format_version": 1,
        "typenet": {"frozen": True, "sha256": typing_audit["encoder_sha256"],
                    "input": "one up-to-50-key sequence [50,5] plus valid length", "output": "one [128] embedding"},
        "face": {"features": list(FEATURES), "training_source_sha256": face_audit["source_sha256"],
                 "compatible_raw_extractor": None,
                 "status": "Trained on supplied UFNet summaries; raw extractor numerical parity unverified"},
        "fusion": {"path": "models/gallery.pt", "sha256": digest, "input": {"face": [None, 42], "typing": [None, 10, 128]},
                   "normalization": "Included in fusion bundle; apply once via inference wrapper",
                   "output": "[N,2] control/impaired; mean of three sigmoid probabilities", "threshold": .5},
    })
    _, final_typing = load_typing(typing_run)
    if final_typing != typing_audit or verify_sources(data_dir) != face_audit["source_sha256"]:
        raise ValueError("Source data changed during experiment")
    return {"output": str(out), "counts": counts,
            "participant_metrics": {role: {name: report["participant"] for name, report in records.items()}
                                    for role, records in reports.items()}}
