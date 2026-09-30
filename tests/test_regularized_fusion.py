import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mcifacenet.fusion_data import load_typing, load_typing_sequences
from mcifacenet.fusion_regularized import RegularizedFusionClassifier, RegularizedFusionNetwork
from mcifacenet.fusion_regularized_train import fit_regularized, sequence_scaling
from mcifacenet.model import FEATURES


def bundle(radius):
    torch.manual_seed(11)
    architecture = {"width": 8, "heads": 2, "dropout": .35, "input_dropout": .2,
                    "modality_dropout": .15, "sequence_radius": radius}
    return {"format_version": 2, "classes": ["control", "impaired"], "threshold": .5,
            "face_features": list(FEATURES), "typing_dimensions": 128, "architecture": architecture,
            "scaling": {"face": {"mean": [0.]*42, "scale": [1.]*42}, "typing": {"mean": [0.]*128, "scale": [1.]*128}},
            "sequence_scaling": {"mean": [0.]*5, "scale": [1.]*5, "clip": 5.},
            "states": [RegularizedFusionNetwork(**architecture).state_dict()]}


@pytest.mark.parametrize("radius", [-1, 4, 8, 50])
def test_regularized_export_padding_invariance_and_finite_gradients(radius, tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(13)
    f, t, s = rng.normal(size=(4, 42)), rng.normal(size=(4, 128)), rng.normal(size=(4, 50, 5))
    lengths = np.array([1, 5, 49, 50])
    model = RegularizedFusionClassifier(bundle(radius))
    p = model.predict_proba(f, t, s, lengths)
    assert np.isfinite(p).all()
    assert np.array_equal(p, model.predict_proba(f, t, s, lengths))
    assert np.allclose(p, model.predict_proba(f, t, s, lengths, batch_size=1), atol=2e-7)
    changed = s.copy()
    changed[np.arange(50)[None, :] >= lengths[:, None]] = 1e9
    assert np.array_equal(p, model.predict_proba(f, t, changed, lengths))
    path = tmp_path / 'model.pt'
    model.export(path)
    assert np.array_equal(p, RegularizedFusionClassifier.load(path).predict_proba(f, t, s, lengths))
    network = RegularizedFusionNetwork(**model.bundle['architecture'])
    network.train()
    tensors = model.prepare(f, t, s, lengths)
    first, second = network(*tensors), network(*tensors)
    assert not torch.equal(first, second)
    first.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters())


def test_local_attention_blocks_distant_keys_and_has_no_fully_masked_query():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    local = RegularizedFusionNetwork(width=8, heads=2, sequence_radius=4).eval()
    full = RegularizedFusionNetwork(width=8, heads=2, sequence_radius=50).eval()
    full.load_state_dict(local.state_dict())
    x = torch.randn(1, 50, 5)
    changed = x.clone()
    changed[:, 20] += 4.
    lengths = torch.tensor([50])
    with torch.inference_mode():
        assert torch.equal(local.sequence_tokens(x, lengths)[:, 0], local.sequence_tokens(changed, lengths)[:, 0])
        assert not torch.allclose(full.sequence_tokens(x, lengths)[:, 0], full.sequence_tokens(changed, lengths)[:, 0])
    mask, valid = local.attention_mask(torch.tensor([1, 5, 50]))
    assert not mask.all(-1).any()
    assert mask[4, 20, 15] and not mask[4, 20, 16]
    assert torch.equal(valid.sum(1), torch.tensor([1, 5, 50]))


def test_sequence_scaling_ignores_padding_and_balances_people():
    sequence = np.zeros((3, 50, 5))
    lengths = np.array([1, 2, 1])
    owners = np.array(['a', 'a', 'b'])
    sequence[0, 0] = 0
    sequence[1, :2] = 2
    sequence[2, 0] = 5
    scale = sequence_scaling(sequence, lengths, owners)
    assert np.allclose(scale['mean'], 3.)  # a mean=1, b mean=5, equal people
    sequence[np.arange(50)[None, :] >= lengths[:, None]] = 1e9
    assert scale == sequence_scaling(sequence, lengths, owners)


@pytest.mark.parametrize('lengths', [[0, 50], [51, 50], [1.5, 50.]])
def test_sequence_inference_rejects_invalid_lengths(lengths):
    model = RegularizedFusionClassifier(bundle(4))
    with pytest.raises(ValueError, match='integer lengths'):
        model.predict_proba(np.zeros((2, 42)), np.zeros((2, 128)), np.zeros((2, 50, 5)), np.asarray(lengths))


def test_early_stopping_uses_validation_loss_and_reloads_selected_weights():
    torch.set_num_threads(1)
    model = RegularizedFusionClassifier(bundle(-1))
    rng = np.random.default_rng(3)
    data = {'tensors': model.prepare(rng.normal(size=(6, 42)), rng.normal(size=(6, 128))),
            'y': np.array([0, 0, 0, 1, 1, 1]), 'owner': np.array(['a', 'a', 'b', 'c', 'c', 'd'])}
    cfg = {'epochs': 6, 'patience': 2, 'min_delta': 100., 'batch_size': 4,
           'learning_rate': .001, 'weight_decay': .1, 'label_smoothing': .05, 'gradient_clip': 1.}
    state, trace = fit_regularized(model.bundle['architecture'], 3, data, data, cfg)
    assert trace['best_epoch'] == 1
    assert len(trace['epochs']) == 3
    assert trace['stopping_reason'] == 'patience'
    assert trace['optimizer_updates'] == 6
    assert all(torch.isfinite(v).all() for v in state.values())


def test_local_sequence_sources_match_existing_embeddings():
    protocol = Path('runs/facial_typing_fusion_fixed_v1/protocol.json')
    if not protocol.exists():
        pytest.skip('Local fixed fusion sources unavailable')
    run = json.loads(protocol.read_text())['typing_run']
    if not Path(run).exists():
        pytest.skip('Registered typing cache unavailable')
    groups, _ = load_typing(run)
    sequence, audit = load_typing_sequences(run, groups)
    assert audit['full_windows'] + audit['partial_windows'] == 12916
    for role, data in sequence.items():
        assert data['sequence'].shape == (len(groups[role]['y']), 50, 5)
        assert (data['lengths'] >= 1).all()
