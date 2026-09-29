"""Isolated exp4 reference scoring with its original float32 preprocessing."""

# On this macOS runtime, importing XGBoost after torch can crash its pickle
# deserializer. The experiment runs this module in a fresh subprocess so native
# initialization order is controlled independently of the training process.
import xgboost
import torch
import hashlib
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning

from .fusion_data import checked, load_typing, sha256


def legacy_normalize(x, state):
    # sklearn 1.7.1 subtracts/divides by float64 scaler parameters in-place.
    # sklearn 1.9 first casts those parameters to float32, shifting some values
    # across XGBoost split boundaries. Preserve exp4's original arithmetic.
    result = np.where(np.isnan(x), state["medians"], x).copy()
    result -= state["scaler"].mean_
    result /= state["scaler"].scale_
    return result.astype(np.float32)


def typing_reference(run, groups):
    """Score with the existing 25-head exp4 classifier and verify recorded parity."""
    run = Path(run).resolve()
    artifact = run.parent / "exports/classifier.joblib"
    verification = json.loads((run.parent / "exports/verification.json").read_text())
    digest = checked(artifact, verification["artifacts"]["classifier.joblib"])
    registration = json.loads((run / "registration.json").read_text())
    # This is a user-owned, hash-verified local artifact, not an arbitrary download.
    with warnings.catch_warnings(record=True) as compatibility:
        warnings.simplefilter("always", InconsistentVersionWarning)
        bundle = joblib.load(artifact)
    engine = bundle["engine"]
    expected = hashlib.sha256((sha256(run / "freeze.json") + registration["encoder_sha256"] + "xgboost").encode()).hexdigest()
    if (bundle["variant"] != "classifier" or engine["fingerprint"] != expected
            or engine["features"] != 128 or engine["family"] != "xgboost" or len(engine["models"]) != 25):
        raise ValueError("Typing export does not match the registered embedding experiment")
    results, by_model = {}, {}
    for role, data in groups.items():
        rows = []
        for model in engine["models"]:
            if (model["name"] != "xgboost" or model["kind"] != "classification"
                    or model.get("encoder") is not None or model["synthetic_rows"] != 0):
                raise ValueError("Unexpected typing reference head")
            state = model["preprocess"]
            normalized = legacy_normalize(data["x"], state)
            rows.append(model["estimator"].predict_proba(normalized)[:, 1])
        by_model[role] = np.stack(rows)
        results[role] = by_model[role].mean(0)
    maximum = 0.0
    with np.load(run / "predictions.npz", allow_pickle=False) as recorded:
        for role in ("validation", "test"):
            for index in np.flatnonzero(recorded["split"] == role):
                owner = recorded["ids"][index]
                reproduced = by_model[role][:, groups[role]["owner"] == owner].mean(1)
                maximum = max(maximum, float(np.max(np.abs(reproduced - recorded["xgboost"][:, index]))))
    if maximum > 1e-6:
        raise ValueError(f"Typing reference predictions changed: {maximum}")
    return results, {"sha256": digest, "fingerprint": expected, "heads": 25,
                     "held_out_participant_prediction_max_difference": maximum,
                     "compatibility_warnings": sorted({str(w.message) for w in compatibility})}


if __name__ == "__main__":
    run, output = Path(sys.argv[1]), Path(sys.argv[2])
    groups, _ = load_typing(run)
    predictions, verification = typing_reference(run, groups)
    np.savez_compressed(output / "typing_reference_predictions.npz", **predictions)
    (output / "typing_reference_verification.json").write_text(
        json.dumps(verification, indent=2, allow_nan=False) + "\n")
