import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mcifacenet import FacialClassifier
from mcifacenet.model import FEATURES, feature_vector


@pytest.fixture
def artifact():
    return {
        "format_version": 1, "features": list(FEATURES), "classes": ["control", "impaired"],
        "mean": [1.0] * 42, "scale": [2.0] * 42,
        "weights": [0.7] + [0.0] * 41, "intercept": -0.1,
        "dropout_probability": 0.10661756438565197, "calibration": None, "threshold": 0.5,
    }


def record(value):
    return dict.fromkeys(FEATURES, value)


def test_feature_order_and_metadata_do_not_change_prediction(artifact):
    model = FacialClassifier(artifact)
    a = {name: index / 10 for index, name in enumerate(FEATURES)}
    b = dict(reversed(list(a.items())))
    b.update(ID="example", pd="yes", Diagnosis="1", date="2020-01-01")
    assert model.predict_proba([a]) == model.predict_proba([b])
    b.update(pd="no", Diagnosis="0", ID="different")
    assert model.predict_proba([a]) == model.predict_proba([b])


@pytest.mark.parametrize("bad", [None, "", "nan", float("inf"), True])
def test_invalid_features_fail_instead_of_becoming_zero(bad):
    row = record(1)
    row[FEATURES[0]] = bad
    with pytest.raises(ValueError, match="finite number"):
        feature_vector(row)


def test_feature_schema_is_explicit():
    row = record(1)
    row.pop(FEATURES[0])
    row["smile_AU1_mean"] = 1
    with pytest.raises(ValueError, match="schema mismatch"):
        feature_vector(row)


def test_prediction_is_batch_and_order_invariant(artifact):
    model = FacialClassifier(artifact)
    rows = [record(-1), record(1), record(10)]
    expected = [model.predict_proba([r])[0] for r in rows]
    assert model.predict_proba(rows) == expected
    assert model.predict_proba(rows[::-1]) == expected[::-1]
    assert all(sum(p) == 1 and 0 <= p[1] <= 1 for p in expected)


def test_exact_dropout_expectation_matches_torch_sampling(artifact):
    torch = pytest.importorskip("torch")
    torch.manual_seed(123)
    torch.set_num_threads(1)
    # Independent stochastic evaluation of the upstream scalar-logit dropout.
    x = torch.tensor([[2.0] * 42])
    normalized = (x - torch.tensor(artifact["mean"])) / torch.tensor(artifact["scale"])
    linear = torch.nn.Linear(42, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([artifact["weights"]]))
        linear.bias.copy_(torch.tensor([artifact["intercept"]]))
        logits = linear(normalized).repeat(200000, 1)
        sampled = torch.sigmoid(torch.nn.functional.dropout(
            logits, p=artifact["dropout_probability"], training=True)).mean().item()
    exact = FacialClassifier(artifact).predict_proba([record(2)])[0][1]
    assert abs(exact - sampled) < 0.002


def test_calibration_and_export_work_without_site_packages(artifact, tmp_path):
    artifact["calibration"] = {"slope": 0.8, "intercept": -0.3}
    model = FacialClassifier(artifact)
    target = tmp_path / "classifier.json"
    model.export(target)
    with pytest.raises(FileExistsError):
        model.export(target)
    assert FacialClassifier.load(target).fingerprint == model.fingerprint
    module = Path(__file__).parents[1] / "src/mcifacenet/model.py"
    code = (
        "import importlib.util,json,sys; "
        "s=importlib.util.spec_from_file_location('portable',sys.argv[1]); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "c=m.FacialClassifier.load(sys.argv[2]); "
        "print(json.dumps(c.predict_proba([dict.fromkeys(m.FEATURES,2)])))"
    )
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(module), str(target)],
                            check=True, text=True, capture_output=True)
    assert json.loads(result.stdout) == model.predict_proba([record(2)])


@pytest.mark.parametrize("field,value", [
    ("classes", ["impaired", "control"]), ("scale", [0] * 42),
    ("weights", [1]), ("dropout_probability", 1), ("threshold", 0),
])
def test_invalid_artifact_is_rejected(artifact, field, value):
    artifact[field] = value
    with pytest.raises(ValueError):
        FacialClassifier(artifact)


def test_changed_source_file_is_rejected(tmp_path, monkeypatch):
    from mcifacenet import data
    source = tmp_path / "features.csv"
    source.write_bytes(b"original")
    manifest = [{"path": source.name, "sha256": hashlib.sha256(b"original").hexdigest()}]
    monkeypatch.setattr(data, "source_manifest", lambda: manifest)
    data.verify_sources(tmp_path)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.verify_sources(tmp_path)


def test_available_source_data_has_disjoint_participants_and_explicit_labels():
    from mcifacenet.data import DEFAULT_DATA, load_datasets
    if not DEFAULT_DATA.exists():
        pytest.skip("Run mcifacenet fetch for the source-data integration check")
    groups, audit = load_datasets(DEFAULT_DATA)
    assert len(audit["excluded"]) == 39
    assert {name: len(rows) for name, rows in groups.items()} == {
        "train": 863, "validation": 340, "calibration": 129, "test": 313, "youtube": 251}
    for left in ("train", "validation", "calibration", "test"):
        for right in ("train", "validation", "calibration", "test"):
            if left != right:
                assert not {r["id"] for r in groups[left]} & {r["id"] for r in groups[right]}


def test_completed_export_scaler_uses_training_rows_only():
    from mcifacenet.data import DEFAULT_DATA, load_datasets
    path = Path("runs/facial_classifier_v1/classifier.json")
    if not path.exists():
        pytest.skip("First experiment has not been trained")
    np = pytest.importorskip("numpy")
    model = FacialClassifier.load(path)
    groups, _ = load_datasets(DEFAULT_DATA)
    x = np.asarray([r["features"] for r in groups["train"]])
    assert np.allclose(model.artifact["mean"], x.mean(axis=0), atol=1e-12, rtol=1e-12)
    assert np.allclose(model.artifact["scale"], np.where(x.std(axis=0) == 0, 1, x.std(axis=0)))
