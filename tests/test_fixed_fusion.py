import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mcifacenet.fusion import FusionNetwork
from mcifacenet.fusion_data import fixed_participant_pairs
from mcifacenet.fusion_fixed import audit_pairs
from mcifacenet.fusion_train import fit_network, network_scores, participant_metrics


def sample():
    rng = np.random.default_rng(17)
    face = {"x": rng.normal(size=(8, 42)), "y": np.array([0, 0, 0, 1, 1, 1, 0, 1]),
            "owner": np.array(["f0", "f0", "f1", "f2", "f2", "f3", "mixed", "mixed"])}
    typing = {"x": rng.normal(size=(6, 128)), "y": np.array([0, 0, 0, 1, 1, 1]),
              "owner": np.array(["t0", "t0", "t1", "t2", "t2", "t3"])}
    return face, typing


def test_fixed_pairs_are_bijective_cover_windows_and_exclude_conflicting_faces():
    face, typing = sample()
    fi, ti = fixed_participant_pairs(face, typing, 23)
    assert np.array_equal(np.sort(ti), np.arange(6))
    assert np.array_equal(face["y"][fi], typing["y"][ti])
    assert "mixed" not in face["owner"][fi]
    for owner in np.unique(typing["owner"]):
        assert len(set(face["owner"][fi][typing["owner"][ti] == owner])) == 1
    for owner in np.unique(face["owner"][fi]):
        assert len(set(typing["owner"][ti][face["owner"][fi] == owner])) == 1
    assert all(np.array_equal(a, b) for a, b in zip((fi, ti), fixed_participant_pairs(face, typing, 23)))
    audit = audit_pairs(face, typing, fi, ti)
    assert audit["participant_pairs"] == 4
    assert audit["window_pairs"] == 6
    assert audit["facial_people_with_conflicting_labels"] == 1


def test_fixed_pairs_fail_instead_of_reusing_people_or_silently_dropping_typists():
    face, typing = sample()
    face["owner"][2] = "f0"
    with pytest.raises(ValueError, match="enough distinct"):
        fixed_participant_pairs(face, typing, 23)
    face, typing = sample()
    typing["y"][0] = 1
    with pytest.raises(ValueError, match="Inconsistent"):
        fixed_participant_pairs(face, typing, 23)


def test_audit_rejects_duplicate_windows_and_shared_partners():
    face, typing = sample()
    fi, ti = fixed_participant_pairs(face, typing, 23)
    bad_ti = ti.copy()
    bad_ti[0] = bad_ti[1]
    with pytest.raises(ValueError, match="exactly once"):
        audit_pairs(face, typing, fi, bad_ti)
    bad_fi = fi.copy()
    bad_fi[typing["y"][ti] == 0] = 0
    with pytest.raises(ValueError, match="exactly one partner"):
        audit_pairs(face, typing, bad_fi, ti)


def test_participant_scores_average_probabilities_before_thresholding():
    # Three windows are one person, not three independent correct/incorrect votes.
    report = participant_metrics([1, 1, 1, 0], [.49, .49, .99, .1], ["a", "a", "a", "b"])
    assert report["n"] == 2
    assert report["accuracy"] == 1
    assert report["recall"] == 1
    with pytest.raises(ValueError, match="conflicting"):
        participant_metrics([0, 1], [.2, .8], ["a", "a"])


def test_fixed_training_visits_every_window_once_per_epoch_and_selects_participant_f1():
    torch.set_num_threads(1)
    face, typing = sample()
    fi, ti = fixed_participant_pairs(face, typing, 23)
    validation = {"face": torch.tensor(face["x"][fi], dtype=torch.float32),
                  "typing": torch.tensor(typing["x"][ti], dtype=torch.float32),
                  "y": typing["y"][ti], "owner": typing["owner"][ti], "draw": np.zeros(len(ti), dtype=int)}
    scaling = {"face": {"mean": [0.] * 42, "scale": [1.] * 42},
               "typing": {"mean": [0.] * 128, "scale": [1.] * 128}}
    cfg = {"width": 8, "heads": 2, "dropout": .1, "epochs": 3, "pairs_per_epoch": 9999,
           "batch_size": 4, "learning_rate": .001, "weight_decay": .01}
    state, trace = fit_network("attention", 123, face, typing, validation, scaling, cfg, fixed_pairs=(fi, ti))
    rng = np.random.default_rng(123)
    for epoch in trace["epochs"]:
        order = rng.permutation(len(ti))
        assert epoch["pair_indices_sha256"] == hashlib.sha256(fi[order].tobytes() + ti[order].tobytes()).hexdigest()
        assert epoch["validation"]["n"] == 4
    assert trace["optimizer_updates"] == 6
    best = max(trace["epochs"], key=lambda e: (e["validation"]["f1"], -e["validation"]["log_loss"]))
    assert best["epoch"] == trace["best_epoch"]
    model = FusionNetwork("attention", width=8, heads=2, dropout=.1)
    model.load_state_dict(state)
    scores = network_scores(model, validation["face"], validation["typing"])
    assert participant_metrics(validation["y"], scores, validation["owner"]) == best["validation"]
