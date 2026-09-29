"""Portable inference: no training framework, pickle, or personal state required."""

import hashlib
import json
import math
from pathlib import Path

FEATURES = tuple(
    f"smile_{measure}_{stat}"
    for measure in (
        "AU01", "AU06", "AU12", "AU14", "AU25", "AU26", "AU45",
        "eye-open-right", "eye-open-left", "eye-raise-right", "eye-raise-left",
        "mouth-open", "mouth-width", "jaw-open",
    )
    for stat in ("mean", "var", "entropy")
)


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def logit(probability):
    value = min(max(probability, 1e-7), 1 - 1e-7)
    return math.log(value / (1 - value))


def feature_vector(record):
    """Read named features; permit metadata but reject changed feature schemas."""
    missing = set(FEATURES) - record.keys()
    extra = {key for key in record if key.startswith("smile_")} - set(FEATURES)
    if missing or extra:
        raise ValueError(f"Feature schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    values = []
    for key in FEATURES:
        value = record[key]
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a finite number")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        values.append(value)
    return values


class FacialClassifier:
    """Return [P(control), P(impaired)] for each feature record."""

    def __init__(self, artifact):
        # Detach caller-owned structures before validation and fingerprinting.
        self.artifact = json.loads(json.dumps(artifact, allow_nan=False))
        a = self.artifact
        if a.get("format_version") != 1 or a.get("features") != list(FEATURES):
            raise ValueError("Unsupported artifact version or feature schema")
        if a.get("classes") != ["control", "impaired"]:
            raise ValueError("Unsupported class order")
        for field in ("mean", "scale", "weights"):
            values = a.get(field)
            if not isinstance(values, list) or len(values) != len(FEATURES):
                raise ValueError(f"Invalid {field} dimension")
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                       and math.isfinite(x) for x in values):
                raise ValueError(f"Nonfinite or nonnumeric {field}")
        if any(x <= 0 for x in a["scale"]):
            raise ValueError("All scales must be positive")
        for field in ("intercept", "dropout_probability", "threshold"):
            value = a.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"Invalid {field}")
        if not 0 <= a["dropout_probability"] < 1 or not 0 < a["threshold"] < 1:
            raise ValueError("Invalid dropout probability or threshold")
        cal = a.get("calibration")
        if cal is not None:
            if set(cal) != {"slope", "intercept"} or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                for x in cal.values()
            ):
                raise ValueError("Invalid calibration parameters")
        self.fingerprint = hashlib.sha256(
            json.dumps(a, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def export(self, path):
        """Refuse to overwrite an existing model."""
        with Path(path).open("x") as stream:
            json.dump(self.artifact, stream, indent=2, allow_nan=False)
            stream.write("\n")

    def predict_proba(self, records):
        output = []
        a = self.artifact
        keep = 1 - a["dropout_probability"]
        for record in records:
            values = feature_vector(record)
            linear = a["intercept"] + math.fsum(
                weight * ((value - mean) / scale)
                for weight, value, mean, scale in zip(a["weights"], values, a["mean"], a["scale"])
            )
            if not math.isfinite(linear):
                raise ValueError("Feature magnitude overflows model arithmetic")
            # UFNet's ShallowANN drops its single output logit. Its MC mean
            # has this exact expectation, so inference needs no random draws.
            probability = keep * sigmoid(linear / keep) + (1 - keep) * 0.5
            if a["calibration"] is not None:
                cal = a["calibration"]
                probability = sigmoid(cal["slope"] * logit(probability) + cal["intercept"])
            output.append([1 - probability, probability])
        return output
