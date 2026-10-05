"""Deterministic 14-channel frame measurements -> 42 temporal statistics.

No camera, image, video, OpenFace, MediaPipe, or filesystem access is needed.
This implements the project's versioned numerical aggregator, not verified
numerical parity with the unreleased UFNet raw feature extractor.
"""

import numpy as np


EXTRACTOR_VERSION = "openface-static_mediapipe478_v1"
AUS = ("AU01", "AU06", "AU12", "AU14", "AU25", "AU26", "AU45")
SIGNALS = AUS + ("eye-open-right", "eye-open-left", "eye-raise-right", "eye-raise-left",
                "mouth-open", "mouth-width", "jaw-open")
FEATURES = tuple(f"smile_{signal}_{stat}" for signal in SIGNALS for stat in ("mean", "var", "entropy"))


def summarize(measurements, au_presence):
    """Return float64[42] from usable measurements[T,14] and AU flags[T,7].

    Each input row must already pass the frontend's frame quality checks.
    Presence is required: intensity alone cannot reproduce active-AU filtering.
    """
    x, presence = np.asarray(measurements, dtype=np.float64), np.asarray(au_presence)
    if (x.ndim != 2 or x.shape[1] != 14 or len(x) < 30 or presence.shape != (len(x), 7)
            or not np.isfinite(x).all() or (x < 0).any() or (x[:, :7] > 5).any()
            or not np.isin(presence, [0, 1]).all()):
        raise ValueError("Expected >=30 usable finite nonnegative frames[T,14], AUs in [0,5], binary presence[T,7]")
    output = []
    for i in range(14):
        values = x[presence[:, i] == 1, i] if i < 7 else x[:, i]
        if not len(values):
            output.extend((0., 0., 0.))
            continue
        total = values.sum()
        p = values[values > 0] / total if total else np.array([])
        entropy = float(-np.sum(p * np.log(p))) if len(p) else 0.
        output.extend((float(values.mean()), float(values.var(ddof=0)), entropy))
    result = np.asarray(output, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Facial summary overflow")
    return result
