import copy
import csv
import json

import numpy as np
import pytest

from mcifacenet.model import FEATURES
from mcifacenet.youtube_train import audit_isolation, averaged_score, load_windows, partition, score, sha256, video_id


def records():
    rows = []
    for split, n in (("train", 24), ("val", 8), ("test", 8), ("", 40)):
        for i in range(n):
            vid = f"source{len(rows)}"
            for window in range(2):
                rows.append({"video_id": vid, "clip_id": f"clip_{vid}", "video_sha256": f"hash_{vid}",
                             "source_split": split, "label": i % 2 if split else 0,
                             "features": [len(rows) + 0.123] * 42})
    return rows


def test_partition_preserves_published_groups_and_is_order_independent():
    rows = records()
    groups = partition(rows, 462)
    reversed_groups = partition(list(reversed(rows)), 462)
    assignment = {r["video_id"]: split for split, group in groups.items() for r in group}
    assert assignment == {r["video_id"]: split for split, group in reversed_groups.items() for r in group}
    assert sum(map(len, groups.values())) == len(rows)
    for row in rows:
        if row["source_split"] == "test":
            assert assignment[row["video_id"]] == "test"
        if row["source_split"] == "val":
            assert assignment[row["video_id"]] == "validation"
        if row["source_split"] == "train":
            assert assignment[row["video_id"]] in ("train", "calibration")
    audit_isolation(groups)


def test_repeated_url_and_overlapping_clips_stay_together():
    rows = records()
    repeat = {**rows[-1], "clip_id": "another_segment", "features": [999] * 42}
    original = partition(rows, 462)
    groups = partition([*rows, repeat], 462)
    assigned = [split for split, group in groups.items() for r in group if r["video_id"] == repeat["video_id"]]
    assert len(set(assigned)) == 1
    assert assigned[0] == next(split for split, group in original.items() if any(r["video_id"] == repeat["video_id"] for r in group))


def test_conflicting_source_labels_or_splits_fail():
    for field, value in (("label", 1), ("source_split", "test")):
        rows = records()
        rows[1][field] = value
        with pytest.raises(ValueError, match="Conflicting"):
            partition(rows, 462)


@pytest.mark.parametrize("field", ["video_id", "clip_id", "video_sha256", "features"])
def test_overlap_audit_rejects_cross_partition_duplicates(field):
    groups = partition(records(), 462)
    groups["test"][0][field] = copy.deepcopy(groups["train"][0][field])
    with pytest.raises(ValueError):
        audit_isolation(groups)


def test_source_id_ignores_url_spelling_and_query_parameters():
    assert video_id("https://youtu.be/Abcdef123_-?t=7") == video_id("https://www.youtube.com/watch?v=Abcdef123_-&t=30")
    with pytest.raises(ValueError):
        video_id("https://example.com/watch?v=Abcdef123_-")


def test_scores_use_positive_class_and_probability_averaging():
    values = score([0, 0, 1, 1], [.1, .7, .8, .2])
    assert (values["accuracy"], values["precision"], values["recall"], values["f1"]) == (.5, .5, .5, .5)
    assert values["confusion_matrix"] == [[1, 1], [1, 1]]
    rows = [{"clip_id": "a", "label": 0}, {"clip_id": "a", "label": 0}, {"clip_id": "b", "label": 1}]
    result = averaged_score(rows, [.6, .2, .7], "clip_id")
    assert result["n"] == 2 and result["accuracy"] == result["f1"] == 1
    assert score([0, 1], [.1, .1])["precision"] == 0


def test_load_windows_excludes_quality_failures_and_checks_source_labels(tmp_path):
    manifest = {"upstream_commit": "test", "records": []}
    rows = []
    for i in range(3):
        label = "y" if i == 0 else "n"
        source = {"clip_id": f"clip{i}", "status": "downloaded", "pd": label, "split_raw": "train",
                  "url": f"https://youtu.be/Abcdef123_{i}", "start_seconds": 12,
                  "expected_duration_seconds": 10, "verification": {"sha256": f"hash{i}"}}
        manifest["records"].append(source)
        rows.append({"clip_id": f"clip{i}", "start_seconds": 0, "end_seconds": 5,
                     "pd": label, "label": int(label == "y"), "source_video_url": source["url"], "source_split": "train",
                     "quality_status": "review_required" if i == 2 else "passes_automatic_checks",
                     "quality_reasons": "test_failure" if i == 2 else "",
                     **dict.fromkeys(FEATURES, "" if i == 2 else .5)})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    (tmp_path / "summary.json").write_text(json.dumps({"source_manifest_sha256": sha256(path),
        "feature_columns": list(FEATURES), "window_seconds": 5, "window_summaries": 3,
        "window_summaries_passing_checks": 2}))
    def save_rows():
        with (tmp_path / "window_features.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    save_rows()
    accepted, rejected, _, _ = load_windows(tmp_path, path)
    assert len(accepted) == 2 and len(rejected) == 1
    assert accepted[0]["source_start_seconds"] == 12
    rows[0]["label"] = 0
    save_rows()
    with pytest.raises(ValueError, match="provenance"):
        load_windows(tmp_path, path)
