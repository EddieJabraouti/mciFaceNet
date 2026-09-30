"""Fixed, exploratory label-paired fusion experiment; no real matched subjects."""

import copy
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler

from .data import load_datasets, verify_sources
from .fusion import FusionClassifier, FusionNetwork
from .fusion_data import PairSampler, checked, face_arrays, load_typing, participant_weights, sha256
from .model import FEATURES, FacialClassifier


SETTINGS = {
    "seeds": [20260929, 20260930, 20260931],
    "epochs": 64, "pairs_per_epoch": 4096, "batch_size": 256,
    "learning_rate": 0.001, "weight_decay": 0.01,
    "width": 32, "heads": 4, "dropout": 0.1,
    "evaluation_draws": 20,
    "pair_seeds": {"validation": 929100, "test": 929200, "youtube": 929300},
    "threshold": 0.5,
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def classification_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if (y.shape != p.shape or not len(y) or not set(y) <= {0, 1}
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()):
        raise ValueError("Invalid classification predictions")
    tn, fp, fn, tp = [int(x) for x in confusion_matrix(y, p >= 0.5, labels=[0, 1]).ravel()]
    ratio = lambda a, b: a / b if b else 0.0
    return {"n": len(y), "positive": int(y.sum()), "accuracy": (tp + tn) / len(y),
            "recall": ratio(tp, tp + fn), "precision": ratio(tp, tp + fp),
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "specificity": ratio(tn, tn + fp),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier_score": float(np.mean((p - y) ** 2)),
            "confusion_matrix": [[tn, fp], [fn, tp]], "threshold": 0.5}


def paired_metrics(y, p, draws):
    rows = [classification_metrics(y[draws == d], p[draws == d]) for d in np.unique(draws)]
    names = ("accuracy", "recall", "precision", "f1", "specificity", "log_loss", "brier_score")
    return {"mean": {k: float(np.mean([r[k] for r in rows])) for k in names},
            "pairing_sd": {k: float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0
                           for k in names},
            "draws": rows, "pooled": classification_metrics(y, p),
            "interpretation": "SD measures random pairing variation, not participant uncertainty"}


def participant_metrics(y, p, owners):
    """Average probabilities within each fixed synthetic participant pair."""
    y, p, owners = np.asarray(y), np.asarray(p), np.asarray(owners)
    if y.shape != p.shape or y.shape != owners.shape:
        raise ValueError("Participant metadata must align with predictions")
    labels, probabilities = [], []
    for owner in np.unique(owners):
        mask = owners == owner
        if len(set(y[mask])) != 1:
            raise ValueError("Participant has conflicting labels")
        labels.append(y[mask][0])
        probabilities.append(p[mask].mean())
    return classification_metrics(labels, probabilities)


def network_scores(model, face, typing):
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(face), 512):
            result.append(model(face[start:start + 512], typing[start:start + 512]).sigmoid().numpy())
    return np.concatenate(result)


def fit_network(kind, seed, train_face, train_typing, validation, scaling, cfg, fixed_pairs=None):
    torch.manual_seed(seed)
    model = FusionNetwork(kind, cfg["width"], cfg["heads"], cfg["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    tensors = []
    for name, data in (("face", train_face), ("typing", train_typing)):
        s = scaling[name]
        tensors.append(torch.tensor((data["x"] - s["mean"]) / s["scale"], dtype=torch.float32))
    sampler = PairSampler(train_face, train_typing)
    rng = np.random.default_rng(seed)
    if fixed_pairs is not None:
        fixed_fi, fixed_ti = fixed_pairs
        weights = participant_weights(train_typing["owner"][fixed_ti]) * len(fixed_ti)
    best, best_state, history = None, None, []
    for epoch in range(1, cfg["epochs"] + 1):
        if fixed_pairs is None:
            fi, ti = sampler.training(cfg["pairs_per_epoch"], rng)
        else:
            order = rng.permutation(len(fixed_ti))
            fi, ti = fixed_fi[order], fixed_ti[order]
            loss_weights = torch.tensor(weights[order], dtype=torch.float32)
        target = torch.tensor(train_face["y"][fi], dtype=torch.float32)
        model.train()
        losses = []
        for start in range(0, len(fi), cfg["batch_size"]):
            batch = slice(start, start + cfg["batch_size"])
            optimizer.zero_grad()
            logits = model(tensors[0][fi[batch]], tensors[1][ti[batch]])
            if fixed_pairs is None:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target[batch])
            else:
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target[batch], reduction="none") * loss_weights[batch]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite fusion training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
        p = network_scores(model, validation["face"], validation["typing"])
        report = paired_metrics(validation["y"], p, validation["draw"])["mean"]
        if "owner" in validation:
            report = participant_metrics(validation["y"], p, validation["owner"])
        score = (report["f1"], -report["log_loss"])
        if best is None or score > best:
            best, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_pair_loss": float(np.mean(losses)),
                        "validation": report,
                        "pair_indices_sha256": hashlib.sha256(fi.tobytes() + ti.tobytes()).hexdigest()})
    return best_state, {"seed": seed, "best_epoch": best_epoch, "epochs": history,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "optimizer_updates": cfg["epochs"] * int(np.ceil(len(fi) / cfg["batch_size"]))}


def typing_reference(run, output):
    # Keep legacy native-library initialization outside the fusion runtime.
    subprocess.run([sys.executable, "-X", "faulthandler", "-m", "mcifacenet.typing_reference",
                    str(Path(run).resolve()), str(Path(output).resolve())], check=True)
    with np.load(Path(output) / "typing_reference_predictions.npz", allow_pickle=False) as cached:
        predictions = {key: cached[key] for key in cached.files}
    verification = json.loads((Path(output) / "typing_reference_verification.json").read_text())
    return predictions, verification


def train_fusion(data_dir, typing_run, face_model, output):
    cfg = copy.deepcopy(SETTINGS)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows, face_audit = load_datasets(data_dir)
    faces = {role: face_arrays(data) for role, data in rows.items()}
    typing, typing_audit = load_typing(typing_run)
    reference = FacialClassifier.load(face_model)
    face_protocol = json.loads(Path(face_model).with_name("protocol.json").read_text())
    if face_protocol["data_audit"]["source_sha256"] != face_audit["source_sha256"]:
        raise ValueError("Facial reference uses different source data")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    source = Path(__file__).parent
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "settings": cfg,
                "experiment": "Artificial same-label facial/typing fusion; no subject-matched multimodal data",
                "face_data": face_audit, "typing_data": typing_audit,
                "face_model_sha256": sha256(face_model), "typing_run": str(Path(typing_run).resolve()),
                "selection": "Per-seed maximum mean validation F1 across fixed pairing draws; log loss tie-break; earliest epoch on exact tie",
                "ensemble": "Equal probability average across all three prespecified seeds; no test-based model selection",
                "training_sampling": "Face person uniformly then recording; same-label typing person uniformly then window; new pairs every epoch",
                "evaluation_sampling": "Every face recording once per draw; uniform same-label typing person then window; 20 fixed draws",
                "calibration": "New fusion probabilities are uncalibrated; threshold fixed at 0.5",
                "preprocessing": "Participant-balanced StandardScaler fitted separately on each modality's training rows",
                "comparators": ["face_only", "typing_only", "mean_probability", "concatenation", "attention"],
                "limitations": [
                    "Pairs use labels to construct artificial cases; scores do not validate real within-person fusion",
                    "Repeated windows/pairs are dependent; pairing SD is not a confidence interval",
                    "UFNet and exp4 held-out data were inspected in prior experiments; this is an exploratory follow-up",
                    "Same-person dependence between modalities is erased by random pairing",
                    "Anonymous cross-source identity overlap cannot be verified",
                    "The frozen TypeNet encoder was pretrained for identity, not this fusion target",
                    "YouTube clip identities are unverified; YouTube fusion reuses typing-test participants",
                    "One typing input is a 50-key window; no 300-key aggregation or longitudinal state is implemented",
                ],
                "source_code_sha256": {p.name: sha256(p) for p in source.glob("*.py")},
                "versions": {name: version(name) for name in ("numpy", "torch", "scikit-learn", "xgboost")}}
    write_json(out / "protocol.json", protocol)
    # Retain the implementation that generated this experiment's artifacts.
    (out / "source").mkdir()
    for path in source.glob("*.py"):
        (out / "source" / path.name).write_bytes(path.read_bytes())
    scaling = {}
    for name, data in (("face", faces["train"]), ("typing", typing["train"])):
        scaler = StandardScaler().fit(data["x"], sample_weight=participant_weights(data["owner"]))
        scaling[name] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    write_json(out / "scaling.json", scaling)
    pairs = {}
    for role in ("validation", "test", "youtube"):
        typed = typing["test" if role == "youtube" else role]
        fi, ti, draw = PairSampler(faces[role], typed).evaluation(cfg["evaluation_draws"], cfg["pair_seeds"][role])
        pairs[role] = (fi, ti, draw)
        np.savez_compressed(out / f"pairs_{role}.npz", face_index=fi, typing_index=ti, draw=draw,
                            face_id=faces[role]["owner"][fi], face_recording=faces[role]["recording"][fi],
                            face_source_row=faces[role]["source_index"][fi], typing_id=typed["owner"][ti],
                            typing_source_index=typed["source_index"][ti], typing_cohort=typed["cohort"][ti],
                            y=faces[role]["y"][fi])
    fi, ti, draw = pairs["validation"]
    validation = {"y": faces["validation"]["y"][fi], "draw": draw}
    for name, raw in (("face", faces["validation"]["x"][fi]), ("typing", typing["validation"]["x"][ti])):
        validation[name] = torch.tensor((raw - scaling[name]["mean"]) / scaling[name]["scale"], dtype=torch.float32)
    models, history, selection = {}, {}, {}
    for kind in ("attention", "concatenation"):
        states, history[kind] = [], []
        for seed in cfg["seeds"]:
            print(f"Training {kind}, seed {seed}", flush=True)
            state, trace = fit_network(kind, seed, faces["train"], typing["train"], validation, scaling, cfg)
            states.append(state)
            history[kind].append(trace)
            print(f"Selected epoch {trace['best_epoch']} using validation F1", flush=True)
        bundle = {"format_version": 1, "classes": ["control", "impaired"], "threshold": 0.5,
                  "face_features": list(FEATURES), "typing_dimensions": 128,
                  "typing_encoder_sha256": typing_audit["encoder_sha256"],
                  "architecture": {"kind": kind, **{k: cfg[k] for k in ("width", "heads", "dropout")}},
                  "scaling": scaling, "states": states, "seeds": cfg["seeds"],
                  "input_unit": "One facial recording summary and one frozen TypeNet 50-key embedding",
                  "pairing": "Trained on artificial same-label pairs; no labels needed for inference"}
        models[kind] = FusionClassifier(bundle)
        digest = models[kind].export(out / "models" / f"{kind}.pt")
        scores = models[kind].predict_proba(faces["validation"]["x"][fi], typing["validation"]["x"][ti])[:, 1]
        selection[kind] = {"sha256": digest, "epochs": [h["best_epoch"] for h in history[kind]],
                           "validation": paired_metrics(validation["y"], scores, draw)["mean"]}
    write_json(out / "training_history.json", history)
    write_json(out / "selection.json", selection)
    freeze = {"protocol_sha256": sha256(out / "protocol.json"), "selection_sha256": sha256(out / "selection.json"),
              "models": {kind: item["sha256"] for kind, item in selection.items()},
              "evaluation_pair_hashes": {role: sha256(out / f"pairs_{role}.npz") for role in pairs}}
    write_json(out / "freeze.json", freeze)
    # Checkpoint and ensemble choices are now frozen before either test evaluation.
    typing_scores, typing_verification = typing_reference(typing_run, out)
    reports, native, parity = {}, {}, {}
    for role, (fi, ti, draw) in pairs.items():
        typed_role = "test" if role == "youtube" else role
        face_score = np.asarray(reference.predict_proba(dict(zip(FEATURES, x)) for x in faces[role]["x"]))[:, 1]
        scores = {"face_only": face_score[fi], "typing_only": typing_scores[typed_role][ti]}
        scores["mean_probability"] = (scores["face_only"] + scores["typing_only"]) / 2
        for kind, model in models.items():
            f, t = faces[role]["x"][fi], typing[typed_role]["x"][ti]
            scores[kind] = model.predict_proba(f, t)[:, 1]
            reloaded = FusionClassifier.load(out / "models" / f"{kind}.pt")
            difference = float(np.max(np.abs(scores[kind] - reloaded.predict_proba(f, t)[:, 1])))
            if difference != 0:
                raise AssertionError("Fusion export changed predictions")
            parity[f"{kind}_{role}"] = difference
        y = faces[role]["y"][fi]
        reports[role] = {name: paired_metrics(y, p, draw) for name, p in scores.items()}
        np.savez_compressed(out / f"predictions_{role}.npz", y=y, draw=draw, **scores)
        native[f"face_{role}"] = classification_metrics(faces[role]["y"], face_score)
    for role in ("validation", "test"):
        people = np.unique(typing[role]["owner"])
        masks = [typing[role]["owner"] == person for person in people]
        native[f"typing_{role}_participants"] = classification_metrics(
            [typing[role]["y"][mask][0] for mask in masks], [typing_scores[role][mask].mean() for mask in masks])
    fi, ti, _ = pairs["test"]
    np.savez_compressed(out / "example_input.npz", face=faces["test"]["x"][fi[:5]], typing=typing["test"]["x"][ti[:5]])
    result = {"experiment": protocol["experiment"], "threshold": 0.5, "positive_class": "impaired",
              "reports": reports, "native_unimodal_metrics": native,
              "typing_reference_verification": typing_verification, "export_max_absolute_difference": parity,
              "counts": {"face": face_audit["splits"], "typing": typing_audit["splits"]},
              "limitations": protocol["limitations"]}
    write_json(out / "metrics.json", result)
    with (out / "comparison.csv").open("w", newline="") as stream:
        fields = ["split", "model", "accuracy", "recall", "precision", "f1"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for role, records in reports.items():
            for name, value in records.items():
                writer.writerow({"split": role, "model": name, **{k: value["mean"][k] for k in fields[2:]}})
    # Catch concurrent source changes instead of silently producing mixed provenance.
    _, final_typing_audit = load_typing(typing_run)
    if final_typing_audit["hashes"] != typing_audit["hashes"]:
        raise ValueError("Typing sources changed during the experiment")
    if verify_sources(data_dir) != face_audit["source_sha256"]:
        raise ValueError("Facial sources changed during the experiment")
    checked(face_model, protocol["face_model_sha256"])
    checked(Path(typing_run).resolve().parent / "exports/classifier.joblib", typing_verification["sha256"])
    return {"output": str(out), "synthetic_pair_metrics": {
        role: {name: {k: value["mean"][k] for k in ("accuracy", "recall", "precision", "f1")}
               for name, value in records.items()} for role, records in reports.items()}}
