import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mcifacenet.fusion import FusionClassifier, FusionNetwork
from mcifacenet.fusion_data import PairSampler, participant_weights, validate_typing
from mcifacenet.fusion_train import classification_metrics, fit_network
from mcifacenet.model import FEATURES


def sample_data():
    face = {"x": np.zeros((4, 42)), "y": np.array([0, 0, 1, 1]),
            "owner": np.array(["f0", "f0", "f1", "f2"])}
    typing = {"x": np.zeros((6, 128)), "y": np.array([0, 0, 0, 1, 1, 1]),
              "owner": np.array(["t0", "t0", "t1", "t2", "t2", "t3"])}
    return face, typing


def test_pairing_preserves_labels_and_is_reproducible():
    face, typing = sample_data()
    sampler = PairSampler(face, typing)
    a = sampler.evaluation(5, 123)
    b = sampler.evaluation(5, 123)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    fi, ti, draw = a
    assert np.array_equal(face["y"][fi], typing["y"][ti])
    for index in range(5):
        assert np.array_equal(fi[draw == index], np.arange(4))
    changed = sampler.evaluation(5, 124)
    assert not np.array_equal(ti, changed[1])


def test_sampling_balances_people_not_their_number_of_windows():
    face, typing = sample_data()
    fi, ti = PairSampler(face, typing).training(30000, np.random.default_rng(4321))
    for owner in np.unique(face["owner"]):
        assert abs(np.mean(face["owner"][fi] == owner) - 1 / 3) < 0.015
    for label in (0, 1):
        owners = typing["owner"][ti[face["y"][fi] == label]]
        for owner in np.unique(owners):
            assert abs(np.mean(owners == owner) - 0.5) < 0.025
    weights = participant_weights(face["owner"])
    assert weights[0] + weights[1] == pytest.approx(weights[2])


def valid_typing():
    owners = np.array(["a", "b", "c", "d", "e", "f"])
    roles = np.repeat(["train", "validation", "test"], 2)
    data = {"x": np.zeros((6, 128)), "y": np.tile([0, 1], 3), "owner": owners,
            "split": roles, "cohort": np.full(6, "example"), "synthetic": np.zeros(6, dtype=bool)}
    splits = {role: owners[roles == role].tolist() for role in np.unique(roles)}
    return data, splits


def test_typing_subject_overlap_fails_before_pairing():
    data, splits = valid_typing()
    validate_typing(data, splits)
    data["owner"][2] = "a"
    splits["validation"][0] = "a"
    with pytest.raises(ValueError, match="leaks across splits"):
        validate_typing(data, splits)


@pytest.mark.parametrize("bad", ["synthetic", "labels", "nonfinite", "dimensions", "assignments"])
def test_typing_cache_contract_rejects_invalid_sources(bad):
    data, splits = valid_typing()
    if bad == "synthetic":
        data["synthetic"][0] = True
    elif bad == "labels":
        data["y"][0] = 3
    elif bad == "nonfinite":
        data["x"][0, 0] = np.nan
    elif bad == "dimensions":
        data["x"] = data["x"][:, :127]
    else:
        splits["test"] = ["unseen"]
    with pytest.raises(ValueError):
        validate_typing(data, splits)


def bundle(kind):
    torch.manual_seed(123)
    network = FusionNetwork(kind, width=8, heads=2, dropout=0.1)
    return {"format_version": 1, "classes": ["control", "impaired"], "threshold": 0.5,
            "face_features": list(FEATURES), "typing_dimensions": 128,
            "architecture": {"kind": kind, "width": 8, "heads": 2, "dropout": 0.1},
            "scaling": {"face": {"mean": [0.] * 42, "scale": [1.] * 42},
                        "typing": {"mean": [0.] * 128, "scale": [1.] * 128}},
            "states": [network.state_dict()]}


@pytest.mark.parametrize("kind", ["attention", "concatenation"])
def test_export_label_free_inference_and_both_modalities(kind, tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(123)
    face, typing = rng.normal(size=(7, 42)), rng.normal(size=(7, 128))
    model = FusionClassifier(bundle(kind))
    p = model.predict_proba(face, typing)
    assert np.allclose(p.sum(axis=1), 1)
    assert np.allclose(p, model.predict_proba(face, typing, batch_size=1), atol=2e-7)
    assert np.allclose(p, model.predict_proba(face[::-1], typing[::-1])[::-1], atol=2e-7)
    assert not np.allclose(p, model.predict_proba(face * 0, typing))
    assert not np.allclose(p, model.predict_proba(face, typing * 0))
    path = tmp_path / "fusion.pt"
    model.export(path)
    assert np.array_equal(p, FusionClassifier.load(path).predict_proba(face, typing))
    with pytest.raises(FileExistsError):
        model.export(path)
    for f, t in ((face[:, :41], typing), (face, typing[:-1]), (face * np.nan, typing)):
        with pytest.raises(ValueError, match="aligned finite arrays"):
            model.predict_proba(f, t)


def test_reported_precision_recall_f1_have_positive_class_semantics():
    result = classification_metrics([0, 0, 0, 1, 1], [0.1, 0.6, 0.8, 0.9, 0.3])
    assert result["confusion_matrix"] == [[1, 2], [1, 1]]
    assert result["accuracy"] == 0.4
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 0.5
    assert result["f1"] == 0.4


def test_small_training_selects_checkpoint_by_validation_only():
    torch.set_num_threads(1)
    face, typing = sample_data()
    rng = np.random.default_rng(6)
    face["x"] = rng.normal(size=face["x"].shape)
    typing["x"] = rng.normal(size=typing["x"].shape)
    f, t, d = PairSampler(face, typing).evaluation(2, 88)
    validation = {"face": torch.tensor(face["x"][f], dtype=torch.float32),
                  "typing": torch.tensor(typing["x"][t], dtype=torch.float32), "y": face["y"][f], "draw": d}
    cfg = {"width": 8, "heads": 2, "dropout": 0.1, "epochs": 3, "pairs_per_epoch": 32,
           "batch_size": 16, "learning_rate": 0.001, "weight_decay": 0.01}
    state, history = fit_network("attention", 123, face, typing, validation, bundle("attention")["scaling"], cfg)
    expected = max(history["epochs"], key=lambda h: (h["validation"]["f1"], -h["validation"]["log_loss"]))
    assert history["best_epoch"] == expected["epoch"]
    assert len({h["pair_indices_sha256"] for h in history["epochs"]}) == 3
    assert history["optimizer_updates"] == 6
    assert state["attention.in_proj_weight"].shape == (24, 8)


def test_completed_fusion_scaling_uses_only_training_people():
    from mcifacenet.data import DEFAULT_DATA, load_datasets
    from mcifacenet.fusion_data import face_arrays, load_typing
    path = Path("runs/facial_typing_fusion_v1")
    if not (path / "metrics.json").exists():
        pytest.skip("First fusion experiment has not completed")
    protocol = json.loads((path / "protocol.json").read_text())
    scaling = json.loads((path / "scaling.json").read_text())
    rows, _ = load_datasets(DEFAULT_DATA)
    typing, _ = load_typing(protocol["typing_run"])
    for name, data in (("face", face_arrays(rows["train"])), ("typing", typing["train"])):
        people = np.unique(data["owner"])
        counts = {p: int(np.sum(data["owner"] == p)) for p in people}
        weights = np.array([1 / counts[p] for p in data["owner"]])
        weights /= weights.sum()
        # StandardScaler accepts sample weights in the input precision. Account
        # for float32 rounding in the cached typing embeddings, then independently
        # recompute the training-only moments with float64 accumulation.
        weights = weights.astype(data["x"].dtype).astype(np.float64)
        x = data["x"].astype(np.float64)
        mean = np.average(x, axis=0, weights=weights)
        variance = np.average((x - mean) ** 2, axis=0, weights=weights)
        assert np.allclose(scaling[name]["mean"], mean, atol=1e-12, rtol=1e-12)
        assert np.allclose(scaling[name]["scale"], np.sqrt(variance), atol=1e-12, rtol=1e-12)
