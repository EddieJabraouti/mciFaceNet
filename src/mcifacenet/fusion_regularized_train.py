"""Prespecified regularization and sliding-attention comparison on fixed pairs."""

import copy
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_datasets, verify_sources
from .fusion import FusionClassifier
from .fusion_data import (checked, face_arrays, load_typing, load_typing_sequences,
                          participant_weights, sha256)
from .fusion_fixed import audit_pairs
from .fusion_regularized import RegularizedFusionClassifier, RegularizedFusionNetwork
from .fusion_train import classification_metrics, participant_metrics, write_json
from .model import FEATURES


CONFIG = {"seeds": [20260929, 20260930, 20260931], "epochs": 48, "patience": 8,
          "min_delta": .0001, "batch_size": 256, "learning_rate": .0005,
          "weight_decay": .1, "label_smoothing": .05, "gradient_clip": 1.}
ARCHITECTURE = {"width": 16, "heads": 2, "dropout": .35, "input_dropout": .2, "modality_dropout": .15}
CANDIDATES = {"regularized": -1, "sequence_full": 50, "sliding_9": 4, "sliding_17": 8}


def sequence_scaling(sequence, lengths, owners):
    """Equal people, equal windows per person, equal valid keys within a window."""
    valid = np.arange(50)[None, :] < lengths[:, None]
    weights = participant_weights(owners)[:, None] * valid / lengths[:, None]
    x, w = sequence.astype(np.float64)[valid], weights[valid]
    mean = np.average(x, axis=0, weights=w)
    variance = np.average((x-mean)**2, axis=0, weights=w)
    scale = np.sqrt(variance)
    scale[scale < 1e-12] = 1.
    return {"mean": mean.tolist(), "scale": scale.tolist(), "clip": 5.}


def scores(model, tensors, batch_size=256):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(*(t[start:start+batch_size] for t in tensors)).sigmoid().numpy()
                               for start in range(0, len(tensors[0]), batch_size)])


def fit_regularized(architecture, seed, train, validation, cfg, network_class=RegularizedFusionNetwork):
    torch.manual_seed(seed)
    model = network_class(**architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    weights = torch.tensor(participant_weights(train["owner"]) * len(train["y"]), dtype=torch.float32)
    target = torch.tensor(train["y"] * (1-cfg["label_smoothing"]) + .5*cfg["label_smoothing"], dtype=torch.float32)
    rng, history = np.random.default_rng(seed), []
    best, stale, best_epoch, best_state = float("inf"), 0, 0, None
    for epoch in range(1, cfg["epochs"] + 1):
        order = rng.permutation(len(target))
        model.train()
        total_loss = 0.
        for start in range(0, len(order), cfg["batch_size"]):
            idx = order[start:start+cfg["batch_size"]]
            optimizer.zero_grad()
            logits = model(*(tensor[idx] for tensor in train["tensors"]))
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, target[idx], reduction="none") * weights[idx]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite regularized training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            optimizer.step()
            total_loss += loss.item() * len(idx)
        report = participant_metrics(validation["y"], scores(model, validation["tensors"]), validation["owner"])
        if report["log_loss"] < best - cfg["min_delta"]:
            best, stale, best_epoch = report["log_loss"], 0, epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": total_loss/len(target), "validation": report})
        if epoch % 8 == 0:
            print(f"  epoch {epoch}: val loss {report['log_loss']:.4f}, F1 {report['f1']:.4f}", flush=True)
        if stale >= cfg["patience"]:
            break
    return best_state, {"seed": seed, "best_epoch": best_epoch, "epochs": history,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "optimizer_updates": len(history)*int(np.ceil(len(target)/cfg["batch_size"])),
                        "stopping_reason": "patience" if stale >= cfg["patience"] else "epoch_budget"}


def train_regularized_fusion(data_dir, typing_run, fixed_run, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = copy.deepcopy(CONFIG)
    fixed_run = Path(fixed_run)
    base_protocol = json.loads((fixed_run / "protocol.json").read_text())
    base_freeze = json.loads((fixed_run / "freeze.json").read_text())
    checked(fixed_run / "protocol.json", base_freeze["protocol_sha256"])
    checked(fixed_run / "selection.json", base_freeze["selection_sha256"])
    rows, face_audit = load_datasets(data_dir)
    typing, typing_audit = load_typing(typing_run)
    sequences, sequence_audit = load_typing_sequences(typing_run, typing)
    if (face_audit["source_sha256"] != base_protocol["face_data"]["source_sha256"]
            or typing_audit["hashes"] != base_protocol["typing_data"]["hashes"]):
        raise ValueError("Fixed experiment uses different source data")
    faces, pairs, counts, raw = {}, {}, {}, {}
    for role in ("train", "validation", "test"):
        faces[role] = face_arrays(rows[role])
        checked(fixed_run / f"pairs_{role}.npz", base_freeze["pair_hashes"][role])
        with np.load(fixed_run / f"pairs_{role}.npz", allow_pickle=False) as d:
            pairs[role] = {k: d[k] for k in d.files}
        fi, ti = pairs[role]["face_index"], pairs[role]["typing_index"]
        counts[role] = audit_pairs(faces[role], typing[role], fi, ti)
        raw[role] = {"face": faces[role]["x"][fi], "typing": typing[role]["x"][ti],
                     "sequence": sequences[role]["sequence"][ti], "lengths": sequences[role]["lengths"][ti],
                     "y": typing[role]["y"][ti], "owner": typing[role]["owner"][ti]}
    baseline_models = {}
    for kind in ("attention", "concatenation"):
        checked(fixed_run / "models" / f"{kind}.pt", base_freeze["models"][kind])
        baseline_models[f"previous_{kind}"] = FusionClassifier.load(fixed_run / "models" / f"{kind}.pt")
    scaling = baseline_models["previous_attention"].bundle["scaling"]
    sequence_scale = sequence_scaling(raw["train"]["sequence"], raw["train"]["lengths"], raw["train"]["owner"])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    source = Path(__file__).parent
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "settings": cfg,
                "architectures": {k: {**ARCHITECTURE, "sequence_radius": v} for k, v in CANDIDATES.items()},
                "experiment": "Regularization and ordered-keystroke sliding attention on unchanged fixed participant pairs",
                "fixed_run": str(fixed_run.resolve()), "fixed_protocol_sha256": sha256(fixed_run / "protocol.json"),
                "typing_run": str(Path(typing_run).resolve()), "face_data": face_audit, "typing_data": typing_audit,
                "sequence_data": sequence_audit, "counts": counts,
                "pair_hashes": base_freeze["pair_hashes"], "previous_model_hashes": base_freeze["models"],
                "selection": "Per-seed participant validation log loss with min_delta and patience; candidate chosen by ensemble participant validation log loss before any test scoring",
                "preprocessing": "Reuse frozen fixed-run face/embedding training scalers; train-only person-balanced valid-keystroke scaler clipped to +/-5; padding excluded",
                "training": "Every existing fixed window pair once per epoch, shuffled only; equal participant loss weighting; smoothed training targets, original evaluation labels",
                "attention": "One local/full layer on original within-window key order; sinusoidal positions; masked padding; mean valid-key pooling added to typing token before two-token fusion",
                "threshold": .5, "positive_class": "impaired", "calibration": "Uncalibrated",
                "limitations": [
                    "Artificial same-label participant pairs are not recordings of the same patient",
                    "One fixed partner assignment; multiple windows within each participant are dependent",
                    "Participant splits are unchanged, but test participants were inspected in earlier experiments",
                    "Both previous-model comparators use the same fixed training participants and windows",
                    "Facial calibration and YouTube data remain excluded; cross-source identities cannot be linked",
                    "Regularization also reduces model width and changes checkpoint selection from F1 to log loss",
                    "Sequence models add measured keystroke features; full versus sliding compares identical inputs and parameter counts",
                    "Local attention is a dense band mask on at most 50 tokens, not an optimized sparse Longformer implementation",
                    "No temporal facial features are available and no cross-window chronology is assumed",
                    "Four prespecified candidates and three seeds; small validation set and previously inspected test participants",
                ],
                "references": ["https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html", "https://arxiv.org/abs/2004.05150"],
                "source_code_sha256": {p.name: sha256(p) for p in source.glob("*.py")},
                "versions": {name: version(name) for name in ("numpy", "torch", "scikit-learn")}}
    write_json(out / "protocol.json", protocol)
    write_json(out / "scaling.json", {**scaling, "sequence": sequence_scale})
    (out / "source").mkdir()
    for path in source.glob("*.py"):
        (out / "source" / path.name).write_bytes(path.read_bytes())
    for role in pairs:
        (out / f"pairs_{role}.npz").write_bytes((fixed_run / f"pairs_{role}.npz").read_bytes())
    models, histories, selection = {}, {}, {}
    for name, architecture in protocol["architectures"].items():
        torch.manual_seed(0)
        bundle = {"format_version": 2, "classes": ["control", "impaired"], "threshold": .5,
                  "face_features": list(FEATURES), "typing_dimensions": 128,
                  "typing_encoder_sha256": typing_audit["encoder_sha256"],
                  "architecture": architecture, "scaling": scaling, "sequence_scaling": sequence_scale,
                  "states": [RegularizedFusionNetwork(**architecture).state_dict()], "seeds": cfg["seeds"],
                  "sequence_features": sequence_audit["features"],
                  "input_unit": "Fixed facial summary, frozen typing embedding, and optional ordered up-to-50-key sequence"}
        processor = RegularizedFusionClassifier(bundle)
        prepared = {role: {"tensors": processor.prepare(**{k: raw[role][k] for k in ("face", "typing", "sequence", "lengths")}),
                           "y": raw[role]["y"], "owner": raw[role]["owner"]} for role in ("train", "validation")}
        states, histories[name] = [], []
        for seed in cfg["seeds"]:
            print(f"Training {name}, seed {seed}", flush=True)
            state, trace = fit_regularized(architecture, seed, prepared["train"], prepared["validation"], cfg)
            states.append(state)
            histories[name].append(trace)
            write_json(out / "training_history.json", histories)
            print(f"Selected epoch {trace['best_epoch']}; stopped at {len(trace['epochs'])}", flush=True)
        bundle["states"] = states
        model = RegularizedFusionClassifier(bundle)
        models[name] = model
        digest = model.export(out / "models" / f"{name}.pt")
        val = raw["validation"]
        p = model.predict_proba(**{k: val[k] for k in ("face", "typing", "sequence", "lengths")})[:, 1]
        selection[name] = {"sha256": digest, "epochs": [t["best_epoch"] for t in histories[name]],
                           "validation": participant_metrics(val["y"], p, val["owner"])}
    selected = min(selection, key=lambda n: selection[n]["validation"]["log_loss"])
    write_json(out / "selection.json", {"selected": selected, "candidates": selection})
    write_json(out / "freeze.json", {"protocol_sha256": sha256(out / "protocol.json"),
               "selection_sha256": sha256(out / "selection.json"), "selected": selected,
               "models": {name: row["sha256"] for name, row in selection.items()},
               "pair_hashes": {role: sha256(out / f"pairs_{role}.npz") for role in pairs}})
    reports, parity = {}, {}
    # Candidate and checkpoint selection are frozen before test predictions.
    for role, data in raw.items():
        predictions = {name: model.predict_proba(data["face"], data["typing"])[:, 1] for name, model in baseline_models.items()}
        for name, model in models.items():
            arguments = {k: data[k] for k in ("face", "typing", "sequence", "lengths")}
            predictions[name] = model.predict_proba(**arguments)[:, 1]
            reloaded = RegularizedFusionClassifier.load(out / "models" / f"{name}.pt")
            difference = float(np.max(np.abs(predictions[name] - reloaded.predict_proba(**arguments)[:, 1])))
            if difference != 0:
                raise AssertionError("Reloaded regularized model changed predictions")
            parity[f"{name}_{role}"] = difference
        reports[role] = {name: {"window": classification_metrics(data["y"], p),
                                "participant": participant_metrics(data["y"], p, data["owner"])} for name, p in predictions.items()}
        np.savez_compressed(out / f"predictions_{role}.npz", y=data["y"], typing_id=data["owner"], **predictions)
    np.savez_compressed(out / "example_input.npz", **{k: raw["test"][k][:5] for k in ("face", "typing", "sequence", "lengths")})
    result = {"experiment": protocol["experiment"], "selected": selected, "threshold": .5, "positive_class": "impaired",
              "reports": reports, "counts": counts, "export_max_absolute_difference": parity, "limitations": protocol["limitations"]}
    write_json(out / "metrics.json", result)
    with (out / "comparison.csv").open("w", newline="") as stream:
        fields = ["split", "unit", "model", "n", "accuracy", "precision", "recall", "f1"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for role, records in reports.items():
            for name, units in records.items():
                for unit, metrics in units.items():
                    writer.writerow({"split": role, "unit": unit, "model": name, **{k: metrics[k] for k in fields[3:]}})
    _, final_typing = load_typing(typing_run)
    _, final_sequences = load_typing_sequences(typing_run, typing)
    if (final_typing != typing_audit or final_sequences != sequence_audit
            or verify_sources(data_dir) != face_audit["source_sha256"]):
        raise ValueError("Source data changed during experiment")
    return {"output": str(out), "selected": selected, "metrics": {role: report[selected] for role, report in reports.items()}}
