import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mcifacenet.cli import main
from mcifacenet.fusion_gallery import GalleryFusionClassifier, GalleryFusionNetwork, build_galleries
from mcifacenet.fusion_gallery_train import assert_disjoint
from mcifacenet.fusion_regularized_train import fit_regularized
from mcifacenet.model import FEATURES


def bundle():
    torch.manual_seed(12)
    architecture = dict(width=16, heads=2, dropout=.35, input_dropout=.2, modality_dropout=.15)
    return {"format_version": 3, "classes": ["control", "impaired"], "threshold": .5,
            "face_features": list(FEATURES), "typing_dimensions": 128, "gallery_size": 10,
            "architecture": architecture,
            "scaling": {"face": {"mean": [.1]*42, "scale": [2.]*42},
                        "typing": {"mean": [.2]*128, "scale": [3.]*128}},
            "states": [GalleryFusionNetwork(**architecture).state_dict()]}


def test_joint_attention_permutation_invariance_export_and_cli(tmp_path):
    torch.set_num_threads(1)
    model = GalleryFusionClassifier(bundle())
    rng = np.random.default_rng(3)
    face, typing = rng.normal(size=(4, 42)), rng.normal(size=(4, 10, 128))
    observed = []
    hook = model.models[0].attention.register_forward_pre_hook(lambda _, args: observed.append(tuple(args[0].shape)))
    p = model.predict_proba(face, typing)
    hook.remove()
    assert observed == [(4, 11, 16)]
    assert np.array_equal(p, model.predict_proba(face, typing))
    assert np.allclose(p, model.predict_proba(face, typing[:, ::-1]), atol=2e-7)
    assert np.allclose(p, model.predict_proba(face, typing, batch_size=1), atol=2e-7)
    changed = typing.copy()
    changed[:, 9] *= 20
    assert not np.allclose(p, model.predict_proba(face, changed), atol=1e-6)
    assert np.allclose(p.sum(1), 1.)
    assert sum(v.numel() for v in model.models[0].parameters()) == 4481
    assert model.predict_proba(np.empty((0, 42)), np.empty((0, 10, 128))).shape == (0, 2)
    path, inputs, output = tmp_path/'model.pt', tmp_path/'input.npz', tmp_path/'output.json'
    model.export(path)
    assert np.array_equal(p, GalleryFusionClassifier.load(path).predict_proba(face, typing))
    np.savez(inputs, face=face, typing=typing)
    main(['fusion-predict', '--model', str(path), '--input', str(inputs), '--output', str(output)])
    assert np.array_equal(p, json.loads(output.read_text())['probabilities'])


@pytest.mark.parametrize('shape', [(2, 128), (2, 9, 128), (2, 11, 128), (2, 10, 127)])
def test_rejects_incompatible_gallery_shapes(shape):
    with pytest.raises(ValueError, match='N,10,128'):
        GalleryFusionClassifier(bundle()).predict_proba(np.zeros((2, 42)), np.zeros(shape))


def test_rejects_nonfinite_inputs_and_wrong_contract():
    model = GalleryFusionClassifier(bundle())
    t = np.zeros((1, 10, 128))
    t[0, 4, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        model.predict_proba(np.zeros((1, 42)), t)
    b = bundle()
    b['gallery_size'] = 9
    with pytest.raises(ValueError, match='exactly ten'):
        GalleryFusionClassifier(b)


def paired_rows():
    counts = [25, 9, 20]
    owners = np.repeat(['t0', 't1', 't2'], counts)
    fi = np.concatenate([np.arange(25) % 2, np.full(9, 2), np.full(20, 3)])
    return dict(typing_id=owners, typing_index=np.arange(54), typing_source_index=np.arange(54)+100,
                typing_cohort=np.full(54, 'test'), face_index=fi, face_id=np.repeat(['f0', 'f1', 'f2'], counts),
                face_source_row=fi+2, face_recording=fi.astype(str), y=np.repeat([0, 0, 1], counts))


def test_gallery_grouping_keeps_partners_unique_windows_and_drops_tails():
    pairs = paired_rows()
    g, audit, excluded = build_galleries(pairs, 5)
    again, audit2, excluded2 = build_galleries(pairs, 5)
    assert all(np.array_equal(g[k], again[k]) for k in g)
    assert audit == audit2 and np.array_equal(excluded, excluded2)
    assert g['typing_index'].shape == (4, 10)
    assert len(np.unique(g['typing_source_index'])) == 40
    assert set(g['typing_source_index'].ravel()).isdisjoint(excluded)
    assert len(excluded) == 14 and audit['participants'] == 2
    assert audit['excluded_participants'] == [{'typing_id': 't1', 'windows': 9, 'label': 0}]
    assert np.all(pairs['typing_id'][g['typing_index']] == g['typing_id'][:, None])
    assert np.all(pairs['face_id'][g['typing_index']] == g['face_id'][:, None])
    order = np.arange(54)[::-1]
    reordered, _, _ = build_galleries({k: v[order] for k, v in pairs.items()}, 5)
    assert all(np.array_equal(g[k], reordered[k]) for k in g)
    invalid = copy.deepcopy(pairs)
    invalid['typing_source_index'][1] = invalid['typing_source_index'][0]
    with pytest.raises(ValueError, match='unique'):
        build_galleries(invalid, 5)
    invalid = copy.deepcopy(pairs)
    invalid['face_id'][0] = 'different'
    with pytest.raises(ValueError, match='one partner'):
        build_galleries(invalid, 5)
    with pytest.raises(ValueError, match='overlap'):
        assert_disjoint({'train': g, 'test': g})


def test_gallery_training_uses_shared_validation_stopping_and_has_finite_gradients():
    torch.set_num_threads(1)
    model = GalleryFusionClassifier(bundle())
    rng = np.random.default_rng(6)
    data = {'tensors': model.prepare(rng.normal(size=(6, 42)), rng.normal(size=(6, 10, 128))),
            'y': np.array([0, 0, 0, 1, 1, 1]), 'owner': np.array(['a', 'a', 'b', 'c', 'c', 'd'])}
    network = GalleryFusionNetwork(**model.bundle['architecture']).train()
    a, b = network(*data['tensors']), network(*data['tensors'])
    assert not torch.equal(a, b)
    a.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
    cfg = dict(epochs=6, patience=2, min_delta=100., batch_size=4, learning_rate=.001,
               weight_decay=.1, label_smoothing=.05, gradient_clip=1.)
    state, trace = fit_regularized(model.bundle['architecture'], 3, data, data, cfg,
                                   network_class=GalleryFusionNetwork)
    assert trace['best_epoch'] == 1 and len(trace['epochs']) == 3
    assert trace['optimizer_updates'] == 6 and trace['parameters'] == 4481
    assert all(torch.isfinite(v).all() for v in state.values())
