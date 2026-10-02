"""Versioned facial measurements and temporal aggregation, without model fitting.

These are explicit reimplementation choices, not a verified copy of UFNet's
unreleased extractor. See docs/facial_extraction.md for definitions and limits.
"""

import math

import numpy as np

from .model import FEATURES

EXTRACTOR_VERSION = "openface-static_mediapipe478_v1"
AUS = ("AU01", "AU06", "AU12", "AU14", "AU25", "AU26", "AU45")
GEOMETRY_PAIRS = {
    "eye-open-right": (159, 145),
    "eye-open-left": (386, 374),
    "eye-raise-right": (105, 159),
    "eye-raise-left": (334, 386),
    "mouth-open": (13, 14),
    "mouth-width": (61, 291),
    "jaw-open": (13, 152),
}
SIGNALS = AUS + tuple(GEOMETRY_PAIRS)


def geometry(landmarks, width, height):
    """3D point-pair distances / inter-iris distance, in consistent pixel units.

    MediaPipe x and z scale with image width; y scales with image height. Left
    and right are anatomical, assuming an unmirrored image. Iris centers: 468/473.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (478, 3) or not np.isfinite(points).all():
        raise ValueError("Expected 478 finite three-dimensional face landmarks")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    points = points * np.array([width, height, width])
    iris_distance = float(np.linalg.norm(points[468] - points[473]))
    if iris_distance < 10:
        raise ValueError("Inter-iris distance is below the 10-pixel quality threshold")
    distances = np.array([np.linalg.norm(points[a] - points[b]) / iris_distance
                          for a, b in GEOMETRY_PAIRS.values()])
    return distances, iris_distance


def bbox_iou(left, right):
    left, right = np.asarray(left), np.asarray(right)
    size = np.maximum(0, np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2]))
    intersection = float(np.prod(size))
    union = float(np.prod(np.maximum(0, left[2:] - left[:2]))
                  + np.prod(np.maximum(0, right[2:] - right[:2])) - intersection)
    return intersection / union if union > 0 else 0.0


def statistics(values):
    """Mean, population variance, amplitude-normalized Shannon entropy (nats).

    Entropy uses p_i=x_i/sum(x); this is not histogram entropy, permutation
    entropy, or entropy rate. It depends on frame count. Empty active-AU sets
    and all-zero vectors return zeros by this version's explicit convention.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("Statistics require a finite, nonnegative signal vector")
    if not len(x):
        return (0.0, 0.0, 0.0)
    total = x.sum()
    p = x[x > 0] / total if total else np.array([])
    entropy = float(-np.sum(p * np.log(p))) if len(p) else 0.0
    return float(x.mean()), float(x.var(ddof=0)), entropy


def aggregate(signals, presence, valid):
    signals = np.asarray(signals, dtype=np.float64)
    presence = np.asarray(presence)
    valid = np.asarray(valid, dtype=bool)
    n = len(valid)
    if signals.shape != (n, 14) or presence.shape != (n, 7):
        raise ValueError("Expected aligned signals[N,14], presence[N,7], valid[N]")
    if not np.any(valid):
        return dict.fromkeys(FEATURES, None), [0] * 14
    if not np.isfinite(signals[valid]).all() or np.any(signals[valid] < 0):
        raise ValueError("A frame marked valid contains invalid signal values")
    if not np.isin(presence[valid], [0, 1]).all():
        raise ValueError("AU presence must be binary on valid frames")
    values, counts = [], []
    for index in range(14):
        keep = valid & (presence[:, index] == 1) if index < 7 else valid
        x = signals[keep, index]
        counts.append(len(x))
        values.extend(statistics(x))
    return dict(zip(FEATURES, values, strict=True)), counts


def windows(duration, seconds, stride):
    """Full contiguous windows [start,end), without padding a partial tail."""
    if not all(math.isfinite(x) and x > 0 for x in (duration, seconds, stride)):
        raise ValueError("Duration, window and stride must be finite and positive")
    count = max(0, math.floor((duration - seconds + 1e-8) / stride) + 1)
    return [(i * stride, i * stride + seconds) for i in range(count)]


def summarize(frames, start, end, clip_has_multiple_faces=False):
    time = frames["timestamp_seconds"]
    selected = (time >= start) & (time < end)
    valid = frames["valid"][selected]
    features, counts = aggregate(frames["signals"][selected], frames["au_presence"][selected], valid)
    total, usable = len(valid), int(valid.sum())
    reasons = []
    if usable < 30:
        reasons.append("fewer_than_30_valid_frames")
    if total == 0 or usable / total < 0.8:
        reasons.append("valid_frame_fraction_below_0.8")
    if clip_has_multiple_faces:
        reasons.append("multiple_faces_detected_in_clip")
    return {
        "start_seconds": start, "end_seconds": end, "frames": total,
        "valid_frames": usable, "valid_fraction": usable / total if total else 0.0,
        "quality_status": "review_required" if reasons else "passes_automatic_checks",
        "quality_reasons": ";".join(reasons),
        "active_au_counts": ";".join(map(str, counts[:7])), **features,
    }
