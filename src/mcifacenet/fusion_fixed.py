"""Fixed one-to-one participant fusion, using each typing window once per epoch."""

import copy
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .data import load_datasets, verify_sources
from .fusion import FusionClassifier
from .fusion_data import (checked, face_arrays, fixed_participant_pairs, load_typing,
                          participant_weights, sha256)
from .fusion_train import (SETTINGS, classification_metrics, fit_network, participant_metrics,
                           typing_reference, write_json)
from .model import FEATURES, FacialClassifier


def audit_pairs(face, typing, fi, ti):
    """Fail closed if a window or either person's identity is shared across pairs."""
    if not np.array_equal(np.sort(ti), np.arange(len(typing["y"]))):
        raise ValueError("Every typing window must appear exactly once")
    if not np.array_equal(face["y"][fi], typing["y"][ti]):
        raise ValueError("Paired labels disagree")
    fowners, towners = face["owner"][fi], typing["owner"][ti]
    for owners, partners in ((fowners, towners), (towners, fowners)):
        if any(len(set(partners[owners == owner])) != 1 for owner in np.unique(owners)):
            raise ValueError("Each participant must have exactly one partner")
    return {"window_pairs": len(ti), "participant_pairs": len(np.unique(towners)),
            "facial_recordings_used": len(np.unique(fi)),
            "positive_participant_pairs": len(np.unique(towners[typing["y"][ti] == 1])),
            "positive_window_pairs": int(typing["y"][ti].sum()),
            "typing_windows_omitted": 0, "typing_windows_reassigned": 0,
            "facial_people_omitted": len(np.unique(face["owner"])) - len(np.unique(fowners)),
            "facial_people_with_conflicting_labels": sum(
                len(set(face["y"][face["owner"] == owner])) != 1 for owner in np.unique(face["owner"]))}


def train_fixed_fusion(data_dir, typing_run, face_model, previous_run, output):
    cfg = copy.deepcopy(SETTINGS)
    cfg["evaluation_draws"] = 1
    cfg["pair_seeds"] = {"train": 929400, "validation": 929500, "test": 929600}
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows, face_audit = load_datasets(data_dir)
    faces = {role: face_arrays(rows[role]) for role in ("train", "validation", "test")}
    typing, typing_audit = load_typing(typing_run)
    reference = FacialClassifier.load(face_model)
    face_protocol = json.loads(Path(face_model).with_name("protocol.json").read_text())
    if face_protocol["data_audit"]["source_sha256"] != face_audit["source_sha256"]:
        raise ValueError("Facial reference uses different data")
    previous_run = Path(previous_run)
    previous_protocol = json.loads((previous_run / "protocol.json").read_text())
    if (previous_protocol["face_data"]["source_sha256"] != face_audit["source_sha256"]
            or previous_protocol["typing_data"]["hashes"] != typing_audit["hashes"]):
        raise ValueError("Previous fusion run uses different data")
    previous_path = previous_run / "models/attention.pt"
    previous_hash = checked(previous_path, json.loads((previous_run / "selection.json").read_text())["attention"]["sha256"])
    previous = FusionClassifier.load(previous_path)
    pairs, counts = {}, {}
    for role in faces:
        fi, ti = fixed_participant_pairs(faces[role], typing[role], cfg["pair_seeds"][role])
        counts[role] = audit_pairs(faces[role], typing[role], fi, ti)
        pairs[role] = (fi, ti)
    cfg["pairs_per_epoch"] = counts["train"]["window_pairs"]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    source = Path(__file__).parent
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "settings": cfg,
        "experiment": "Fixed one-to-one same-label participant fusion; every real typing window used once per epoch",
        "face_data": face_audit, "typing_data": typing_audit, "counts": counts,
        "typing_run": str(Path(typing_run).resolve()), "face_model_sha256": sha256(face_model),
        "previous_attention_sha256": previous_hash, "previous_run": str(previous_run.resolve()),
        "pairing": "Uniform same-label facial people without replacement; all of a typist's windows stay with one facial person; shuffled recordings cycled within that person, then frozen",
        "training": "All fixed pairs once per epoch, order shuffled only; inverse window-count loss weights give each typing participant equal total weight",
        "preprocessing": "Training-only participant-balanced scalers; only selected facial recordings and all training typing windows",
        "selection": "Maximum validation participant F1 after averaging each pair's window probabilities; log loss tie-break; earliest exact tie",
        "ensemble": "Equal average of the three prespecified seed probabilities; same fixed pairing for every seed and model",
        "reporting": "Participant metrics average probabilities over each fixed pair; window metrics are reported separately",
        "threshold": 0.5, "calibration": "Fusion probabilities are uncalibrated",
        "comparators": ["face_only", "typing_only", "mean_probability", "previous_attention", "concatenation", "attention"],
        "limitations": [
            "Artificial same-label people are not real matched patients; labels are used only to construct the pairs",
            "One fixed assignment; training seeds do not estimate uncertainty from alternative partner assignments",
            "Multiple windows and repeated facial features within each participant pair are dependent",
            "Participant splits remain unchanged; these test subjects were already inspected in earlier experiments",
            "Prior attention and existing unimodal comparators used their original training pools, including more facial people",
            "Matching restricts facial participants; class prevalence, training loss weighting and validation selection differ from the original run",
            "Facial calibration and YouTube data remain excluded; cross-source identity overlap cannot be verified",
            "The TypeNet encoder is frozen; this is a small two-token attention network, not a raw sequence transformer",
        ],
        "source_code_sha256": {p.name: sha256(p) for p in source.glob("*.py")},
        "versions": {name: version(name) for name in ("numpy", "torch", "scikit-learn", "xgboost")},
    }
    write_json(out / "protocol.json", protocol)
    (out / "source").mkdir()
    for path in source.glob("*.py"):
        (out / "source" / path.name).write_bytes(path.read_bytes())
    for role, (fi, ti) in pairs.items():
        np.savez_compressed(out / f"pairs_{role}.npz", face_index=fi, typing_index=ti,
                            face_id=faces[role]["owner"][fi], face_recording=faces[role]["recording"][fi],
                            face_source_row=faces[role]["source_index"][fi], typing_id=typing[role]["owner"][ti],
                            typing_source_index=typing[role]["source_index"][ti],
                            typing_cohort=typing[role]["cohort"][ti], y=typing[role]["y"][ti])
    scaling = {}
    fi, ti = pairs["train"]
    for name, data, index in (("face", faces["train"], np.unique(fi)), ("typing", typing["train"], np.unique(ti))):
        scaler = StandardScaler().fit(data["x"][index], sample_weight=participant_weights(data["owner"][index]))
        scaling[name] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    write_json(out / "scaling.json", scaling)
    fi, ti = pairs["validation"]
    validation = {"y": typing["validation"]["y"][ti], "owner": typing["validation"]["owner"][ti],
                  "draw": np.zeros(len(ti), dtype=int)}
    for name, x in (("face", faces["validation"]["x"][fi]), ("typing", typing["validation"]["x"][ti])):
        validation[name] = torch.tensor((x - scaling[name]["mean"]) / scaling[name]["scale"], dtype=torch.float32)
    models, history, selection = {}, {}, {}
    for kind in ("attention", "concatenation"):
        states, history[kind] = [], []
        for seed in cfg["seeds"]:
            print(f"Training fixed-pair {kind}, seed {seed}", flush=True)
            state, trace = fit_network(kind, seed, faces["train"], typing["train"], validation, scaling, cfg,
                                       fixed_pairs=pairs["train"])
            states.append(state)
            history[kind].append(trace)
            write_json(out / "training_history.json", history)
            print(f"Selected epoch {trace['best_epoch']} by validation participant F1", flush=True)
        bundle = {"format_version": 1, "classes": ["control", "impaired"], "threshold": 0.5,
                  "face_features": list(FEATURES), "typing_dimensions": 128,
                  "typing_encoder_sha256": typing_audit["encoder_sha256"],
                  "architecture": {"kind": kind, **{k: cfg[k] for k in ("width", "heads", "dropout")}},
                  "scaling": scaling, "states": states, "seeds": cfg["seeds"],
                  "input_unit": "One facial recording summary and one frozen TypeNet 50-key embedding",
                  "pairing": protocol["pairing"]}
        models[kind] = FusionClassifier(bundle)
        digest = models[kind].export(out / "models" / f"{kind}.pt")
        p = models[kind].predict_proba(faces["validation"]["x"][fi], typing["validation"]["x"][ti])[:, 1]
        selection[kind] = {"sha256": digest, "epochs": [h["best_epoch"] for h in history[kind]],
                           "validation": participant_metrics(validation["y"], p, validation["owner"])}
    write_json(out / "selection.json", selection)
    write_json(out / "freeze.json", {"protocol_sha256": sha256(out / "protocol.json"),
               "selection_sha256": sha256(out / "selection.json"),
               "models": {kind: item["sha256"] for kind, item in selection.items()},
               "pair_hashes": {role: sha256(out / f"pairs_{role}.npz") for role in pairs}})
    # No test scores are calculated until the selection is saved and frozen.
    typing_scores, typing_verification = typing_reference(typing_run, out)
    reports, parity = {}, {}
    for role, (fi, ti) in pairs.items():
        f, t = faces[role]["x"][fi], typing[role]["x"][ti]
        scores = {"face_only": np.asarray(reference.predict_proba(dict(zip(FEATURES, x)) for x in f))[:, 1],
                  "typing_only": typing_scores[role][ti], "previous_attention": previous.predict_proba(f, t)[:, 1]}
        scores["mean_probability"] = (scores["face_only"] + scores["typing_only"]) / 2
        for kind, model in models.items():
            scores[kind] = model.predict_proba(f, t)[:, 1]
            reloaded = FusionClassifier.load(out / "models" / f"{kind}.pt")
            delta = float(np.max(np.abs(scores[kind] - reloaded.predict_proba(f, t)[:, 1])))
            if delta != 0:
                raise AssertionError("Fusion export changed predictions")
            parity[f"{kind}_{role}"] = delta
        y, owners = typing[role]["y"][ti], typing[role]["owner"][ti]
        reports[role] = {name: {"window": classification_metrics(y, p),
                                "participant": participant_metrics(y, p, owners)} for name, p in scores.items()}
        np.savez_compressed(out / f"predictions_{role}.npz", y=y, typing_id=owners, **scores)
    fi, ti = pairs["test"]
    np.savez_compressed(out / "example_input.npz", face=faces["test"]["x"][fi[:5]], typing=typing["test"]["x"][ti[:5]])
    result = {"experiment": protocol["experiment"], "threshold": 0.5, "positive_class": "impaired",
              "reports": reports, "counts": counts, "export_max_absolute_difference": parity,
              "typing_reference_verification": typing_verification, "limitations": protocol["limitations"]}
    write_json(out / "metrics.json", result)
    with (out / "comparison.csv").open("w", newline="") as stream:
        fields = ["split", "unit", "model", "n", "accuracy", "recall", "precision", "f1"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for role, models_report in reports.items():
            for name, units in models_report.items():
                for unit, metrics in units.items():
                    writer.writerow({"split": role, "unit": unit, "model": name, **{k: metrics[k] for k in fields[3:]}})
    _, final_typing = load_typing(typing_run)
    if final_typing["hashes"] != typing_audit["hashes"] or verify_sources(data_dir) != face_audit["source_sha256"]:
        raise ValueError("Source data changed during the experiment")
    checked(face_model, protocol["face_model_sha256"])
    checked(previous_path, previous_hash)
    checked(Path(typing_run).resolve().parent / "exports/classifier.joblib", typing_verification["sha256"])
    return {"output": str(out), "counts": counts, "attention": {role: report["attention"] for role, report in reports.items()}}
