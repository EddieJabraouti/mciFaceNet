import math
import csv

import numpy as np
import pytest

from mcifacenet.facial_signals import aggregate, bbox_iou, geometry, statistics, summarize, windows
from mcifacenet.model import FEATURES, feature_vector
from mcifacenet.facial_signals import AUS
from mcifacenet.video_extract import read_openface


def test_au_statistics_use_presence_and_validity_independently():
    signals = np.ones((4, 14))
    signals[:, 0] = [1, 3, 5, 4]
    presence = np.ones((4, 7))
    presence[2, 0] = 0
    presence[:, 1] = 0
    values, counts = aggregate(signals, presence, [True, True, True, False])
    assert values["smile_AU01_mean"] == 2
    assert values["smile_AU01_var"] == 1
    assert values["smile_AU01_entropy"] == pytest.approx(-.25 * math.log(.25) - .75 * math.log(.75))
    assert counts[0] == 2
    assert counts[1] == 0
    assert all(values[f"smile_AU06_{stat}"] == 0 for stat in ("mean", "var", "entropy"))
    assert len(feature_vector(values)) == 42
    assert tuple(values) == FEATURES


def test_failed_tracking_does_not_become_a_zero_face():
    values, counts = aggregate(np.full((3, 14), np.nan), np.full((3, 7), np.nan), [False] * 3)
    assert all(value is None for value in values.values())
    assert counts == [0] * 14
    with pytest.raises(ValueError, match="finite"):
        feature_vector(values)


def test_entropy_definition_is_explicit_and_frame_count_dependent():
    assert statistics([2, 2, 2, 2]) == pytest.approx((2, 0, math.log(4)))
    assert statistics([0, 0]) == (0, 0, 0)
    assert statistics([]) == (0, 0, 0)
    assert statistics([1, 3])[2] == pytest.approx(statistics([10, 30])[2])
    for bad in ([1, np.nan], [1, -1], [np.inf]):
        with pytest.raises(ValueError):
            statistics(bad)


def test_geometry_uses_consistent_units_and_inter_iris_normalization():
    points = np.zeros((478, 3))
    points[468] = [.3, .4, 0]
    points[473] = [.7, .4, 0]
    points[159] = [.3, .3, 0]
    points[145] = [.3, .4, 0]
    measurements, distance = geometry(points, 100, 200)
    assert distance == pytest.approx(40)
    assert measurements[0] == pytest.approx(20 / 40)
    scaled, _ = geometry(points + [1, 2, 3], 300, 600)
    np.testing.assert_allclose(scaled, measurements, atol=1e-12)
    points[473] = points[468]
    with pytest.raises(ValueError, match="Inter-iris"):
        geometry(points, 100, 200)


def test_face_match_requires_overlapping_boxes():
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0
    assert bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_windows_have_fixed_boundaries_without_padded_tails():
    assert windows(10, 5, 5) == [(0, 5), (5, 10)]
    assert windows(9, 5, 5) == [(0, 5)]
    assert windows(2, 5, 5) == []
    assert windows(10, 5, 2.5) == [(0, 5), (2.5, 7.5), (5, 10)]
    for args in [(10, 0, 5), (10, 5, -1), (float("nan"), 5, 5)]:
        with pytest.raises(ValueError):
            windows(*args)


def test_window_aggregation_excludes_outside_samples_and_flags_ambiguity():
    signals = np.ones((100, 14))
    signals[50:] = 4
    frames = {"timestamp_seconds": np.arange(100) / 10,
              "signals": signals, "au_presence": np.ones((100, 7)), "valid": np.ones(100, bool)}
    first = summarize(frames, 0, 5)
    second = summarize(frames, 5, 10)
    assert first["smile_AU01_mean"] == 1
    assert second["smile_AU01_mean"] == 4
    assert first["frames"] == second["frames"] == 50
    assert first["quality_status"] == "passes_automatic_checks"
    assert summarize(frames, 0, 5, True)["quality_status"] == "review_required"
    frames["valid"][:30] = False
    assert summarize(frames, 0, 5)["quality_status"] == "review_required"


def test_invalid_valid_frame_fails_instead_of_silently_imputing():
    signals = np.ones((2, 14)); signals[0, 0] = np.nan
    with pytest.raises(ValueError):
        aggregate(signals, np.ones((2, 7)), [True, True])
    with pytest.raises(ValueError):
        aggregate(np.ones((2, 14)), np.full((2, 7), 2), [True, True])


def write_openface_fixture(path, frame_numbers, timestamps):
    rows = []
    for frame, timestamp in zip(frame_numbers, timestamps, strict=True):
        row = {"frame": frame, "timestamp": timestamp, "confidence": .95, "success": 1}
        row.update({f"{au}_{suffix}": 1 for au in AUS for suffix in ("r", "c")})
        row.update({f"{axis}_{i}": i for axis in ("x", "y") for i in range(68)})
        rows.append(row)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_openface_alignment_keeps_missing_frame_invalid(tmp_path):
    path = tmp_path / "openface.csv"
    write_openface_fixture(path, [1, 3], [0, .08])
    frames = read_openface(path, np.array([0, .04, .08]))
    assert frames["openface_success"].tolist() == [True, False, True]
    assert np.isnan(frames["au_intensity"][1]).all()
    assert frames["au_intensity"][2].tolist() == [1] * 7
    assert frames["openface_bbox"][0].tolist() == [0, 0, 67, 67]


@pytest.mark.parametrize("numbers,timestamps", [
    ([1, 1], [0, 0]), ([1, 4], [0, .12]), ([1, 2], [0, 10]), ([1, 2], [0, float("nan")]),
])
def test_openface_rejects_ambiguous_frame_alignment(tmp_path, numbers, timestamps):
    path = tmp_path / "openface.csv"
    write_openface_fixture(path, numbers, timestamps)
    with pytest.raises(ValueError):
        read_openface(path, np.array([0, .04, .08]))
