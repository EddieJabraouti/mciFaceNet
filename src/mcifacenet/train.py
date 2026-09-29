"""Fixed classification experiment; evaluation labels never choose the model."""

import copy
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, log_loss, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import COMMIT, load_datasets
from .model import FEATURES, FacialClassifier, logit


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def as_arrays(rows):
    return np.asarray([r["features"] for r in rows], dtype=np.float64), np.asarray([r["label"] for r in rows])


def records_from_rows(rows):
    return [dict(zip(FEATURES, row["features"])) for row in rows]


def probabilities(model, rows):
    return np.asarray(model.predict_proba(records_from_rows(rows)))[:, 1]


def metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    predicted = scores >= 0.5
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    ece = 0.0
    bins = np.minimum((scores * 10).astype(int), 9)
    for index in range(10):
        selected = bins == index
        if selected.any():
            ece += selected.mean() * abs(scores[selected].mean() - labels[selected].mean())
    return {
        "n": len(labels), "positive": int(labels.sum()),
        "auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
        "average_precision": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "brier_score": float(brier_score_loss(labels, scores)),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "ece_10_equal_width_bins": float(ece),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": 0.5,
    }


def make_artifact(name, scaler, weights, intercept, dropout):
    return {
        "format_version": 1, "model_name": name, "classes": ["control", "impaired"],
        "features": list(FEATURES), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "weights": np.asarray(weights, dtype=float).reshape(-1).tolist(), "intercept": float(intercept),
        "dropout_probability": float(dropout), "calibration": None, "threshold": 0.5,
        "source_commit": COMMIT,
        "training_target": "Explicit PD yes/no labels mapped to impaired/control; no separate diagnostic subtype",
        "input_unit": "42 summary features from one facial recording",
    }


def fit_ufnet(x_train, y_train, validation, scaler, cfg):
    """Adapt the published scalar-logit dropout model using the fixed recipe."""
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    augmented_x, augmented_y = SMOTE(random_state=cfg["random_state"]).fit_resample(x_train, y_train)
    linear = nn.Linear(len(FEATURES), 1)
    model = nn.Sequential(linear, nn.Dropout(cfg["dropout_prob"]), nn.Sigmoid())
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"],
                                momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    loader = DataLoader(TensorDataset(torch.tensor(augmented_x, dtype=torch.float32),
                                     torch.tensor(augmented_y, dtype=torch.float32)),
                        batch_size=cfg["batch_size"], shuffle=True)
    criterion = nn.BCELoss()
    best_loss, best_model, history = float("inf"), None, []
    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x).reshape(-1), y)
            loss.backward()
            optimizer.step()
        artifact = make_artifact("ufnet_shallow_dropout", scaler,
                                 linear.weight.detach().numpy(), linear.bias.detach().item(), cfg["dropout_prob"])
        candidate = FacialClassifier(artifact)
        scores = probabilities(candidate, validation)
        current_loss = float(log_loss([r["label"] for r in validation], scores))
        history.append({"epoch": epoch, "validation_log_loss": current_loss})
        if current_loss < best_loss:
            best_loss, best_model = current_loss, candidate
            best_epoch = epoch
    return best_model, {"best_epoch": best_epoch, "epochs": history,
                        "real_train_rows": len(y_train), "train_rows_after_smote": len(augmented_y)}


def calibrate(model, calibration_rows):
    raw = probabilities(model, calibration_rows)
    x = np.asarray([logit(p) for p in raw]).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    calibrator.fit(x, [r["label"] for r in calibration_rows])
    artifact = copy.deepcopy(model.artifact)
    artifact["calibration"] = {"slope": float(calibrator.coef_[0, 0]),
                               "intercept": float(calibrator.intercept_[0])}
    return FacialClassifier(artifact)


def participant_metrics(rows, scores, seed):
    """Aggregate repeated recordings without inventing a label for changing IDs."""
    groups = {}
    for row, score in zip(rows, scores):
        groups.setdefault(row["id"], []).append((row["label"], score))
    eligible = [v for v in groups.values() if len({label for label, _ in v}) == 1]
    labels = np.asarray([v[0][0] for v in eligible])
    averages = np.asarray([np.mean([p for _, p in v]) for v in eligible])
    result = metrics(labels, averages)
    result["excluded_ids_with_changing_labels"] = len(groups) - len(eligible)
    result["aggregation"] = "arithmetic mean of recording probabilities per participant"
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(1000):
        indices = rng.integers(0, len(labels), len(labels))
        if len(set(labels[indices])) == 2:
            aucs.append(roc_auc_score(labels[indices], averages[indices]))
    result["auroc_95_percentile_bootstrap_interval"] = np.quantile(aucs, [0.025, 0.975]).tolist()
    result["bootstrap_unit"] = "participant"
    return result


def train(data_dir, output):
    groups, audit = load_datasets(data_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    cfg_path = Path(data_dir) / "models/facial_expression_smile_best_auroc_baal/predictive_model/model_config.json"
    cfg = json.loads(cfg_path.read_text())
    if (cfg["model"], cfg["optimizer"], cfg["scaling_method"], cfg["drop_correlated"], cfg["use_scheduler"]) != (
        "ShallowANN", "SGD", "StandardScaler", "no", "no"
    ):
        raise ValueError("Reference configuration no longer matches this implementation")
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "data_audit": audit,
        "primary_selection": "lowest raw validation log loss; neural epoch chosen on the same validation set",
        "calibration": "Platt scaling on published calibration participants, excluded from fitting and selection",
        "threshold": 0.5, "ufnet_configuration": cfg,
        "logistic_reference": {"C": 1.0, "max_iter": 2000, "solver": "lbfgs", "augmentation": "none"},
        "changes_from_upstream": [
            "Use only explicit yes/no labels; exclude 39 ambiguous/missing records",
            "Reserve the published calibration IDs from training",
            "Use the exact expectation of scalar-logit dropout instead of 1000 Monte Carlo trials",
            "Add independent probability calibration and a logistic-regression reference",
            "Evaluate all YouTubePD updated-feature rows as a separate dataset; no YouTube fitting",
        ],
        "limitations": [
            "Adapted facial-only experiment, not exact reproduction of multimodal paper results",
            "Participant independence established within UFNet using author IDs",
            "YouTubePD clip IDs are not subject IDs; cross-dataset identity overlap is unverified",
            "No raw-video feature extraction or longitudinal change validation",
        ],
        "versions": {name: version(name) for name in ("numpy", "scikit-learn", "imbalanced-learn", "torch")},
        "source_code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob("*.py")},
    }
    write_json(output / "protocol.json", protocol)
    x_train, y_train = as_arrays(groups["train"])
    scaler = StandardScaler().fit(x_train)
    scaled_train = scaler.transform(x_train)
    ufnet, history = fit_ufnet(scaled_train, y_train, groups["validation"], scaler, cfg)
    logistic = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    logistic.fit(scaled_train, y_train)
    reference = FacialClassifier(make_artifact("logistic_reference", scaler,
                                                logistic.coef_, logistic.intercept_[0], 0))
    candidates = {"ufnet_shallow_dropout": ufnet, "logistic_reference": reference}
    selection = {name: metrics([r["label"] for r in groups["validation"]],
                               probabilities(model, groups["validation"])) for name, model in candidates.items()}
    selected = min(selection, key=lambda key: selection[key]["log_loss"])
    # Freeze candidate/epoch choice before calibration or either test evaluation.
    write_json(output / "selection.json", {"selected": selected, "validation": selection})
    write_json(output / "training_history.json", history)
    calibrated = {name: calibrate(model, groups["calibration"]) for name, model in candidates.items()}
    reports = {}
    prediction_rows = []
    for name, model in calibrated.items():
        reports[name] = {}
        model.export(output / "models" / f"{name}.json")
        for split, rows in groups.items():
            labels = [r["label"] for r in rows]
            scores = probabilities(model, rows)
            raw_scores = probabilities(candidates[name], rows)
            reports[name][split] = {"calibrated": metrics(labels, scores), "raw": metrics(labels, raw_scores)}
            for row, score, raw in zip(rows, scores, raw_scores):
                prediction_rows.append({"model": name, "split": split, "id": row["id"],
                                        "recording": row["recording"], "label": row["label"],
                                        "p_impaired": float(score), "raw_p_impaired": float(raw)})
    final = calibrated[selected]
    final.export(output / "classifier.json")
    reloaded = FacialClassifier.load(output / "classifier.json")
    all_rows = [row for rows in groups.values() for row in rows]
    difference = float(np.max(np.abs(probabilities(final, all_rows) - probabilities(reloaded, all_rows))))
    if difference != 0:
        raise AssertionError("Export changed predictions")
    prior = float(y_train.mean())
    report = {"selected_model": selected, "models": reports,
              "test_participant_metrics": participant_metrics(groups["test"], probabilities(final, groups["test"]), cfg["seed"]),
              "constant_training_prior": {split: metrics([r["label"] for r in groups[split]],
                                                        np.full(len(groups[split]), prior)) for split in ("test", "youtube")},
              "export_verification": {"recordings": len(all_rows), "max_absolute_prediction_difference": difference,
                                      "model_fingerprint": reloaded.fingerprint},
              "interpretation": {"train": "fitting performance", "validation": "model/epoch selection",
                                 "calibration": "calibrator fitting performance", "test": "held-out UFNet recordings",
                                 "youtube": "separate clip-level dataset; identity overlap unverified"}}
    write_json(output / "metrics.json", report)
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    return {"selected_model": selected, "test": reports[selected]["test"]["calibrated"],
            "youtube": reports[selected]["youtube"]["calibrated"], "output": str(output)}
