from pathlib import Path
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from exports.facial import FEATURES, summarize
from exports.typenet import FrozenTypeNet
from exports.fusion import FrozenFusion
from mcifacenet.facial_signals import aggregate
from mcifacenet.fusion_gallery import GalleryFusionClassifier


def test_numeric_facial_export_exactly_matches_versioned_aggregation():
    rng = np.random.default_rng(73)
    x = rng.uniform(0, 5, (100, 14))
    presence = rng.integers(0, 2, (100, 7))
    presence[:, 3] = 0
    expected, _ = aggregate(x, presence, np.ones(100, dtype=bool))
    assert np.array_equal(summarize(x, presence), [expected[k] for k in FEATURES])
    assert np.array_equal(summarize(x, presence)[9:12], [0, 0, 0])
    for a, b in [(x[:29], presence[:29]), (x, np.full((100, 7), .5)), (x*np.nan, presence)]:
        with pytest.raises(ValueError):
            summarize(a, b)


def test_typenet_event_adapter_units_mapping_lookahead_and_frozen_inference():
    torch.set_num_threads(1)
    model = FrozenTypeNet()
    events = [dict(key='a', press_ms=100*i, release_ms=100*i+120) for i in range(51)]
    seq, lengths = model.prepare_events(events)
    assert lengths.tolist() == [50, 1]
    assert seq[0, 49] == pytest.approx([.12, -.02, .1, .1, 65/255])
    assert seq[1, 0] == pytest.approx([.12, 0, 0, 0, 65/255])
    output = model.encode(seq, lengths)
    assert output.shape == (2, 128) and np.isfinite(output).all()
    changed = seq.copy()
    changed[1, 1:] = 1e6
    assert np.array_equal(output, model.encode(changed, lengths))
    assert all(not p.requires_grad for p in model._encoder.parameters())
    assert not model._encoder.training
    assert len(model.prepare_events(events+events)[0]) == 2
    with pytest.raises(ValueError):
        model.encode(seq, np.array([0, 50]))


def test_export_gallery_parity_and_independent_copy(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(9)
    face, typing = rng.normal(size=(3, 42)), rng.normal(size=(3, 10, 128))
    module = FrozenFusion()
    original = GalleryFusionClassifier.load('src/exports/weights/fusion_gallery.pt')
    expected = original.predict_proba(face, typing)
    assert np.array_equal(expected, module.predict_proba(face, typing))
    shutil.copytree('src/exports', tmp_path/'exports', ignore=shutil.ignore_patterns('__pycache__'))
    np.savez(tmp_path/'input.npz', face=face, typing=typing)
    # A copied package, outside the repository, with no mcifacenet import.
    script = """
import json, sys, numpy as np
from exports.fusion import FrozenFusion
from exports.typenet import FrozenTypeNet
from exports.facial import summarize
d=np.load('input.npz')
assert 'mcifacenet' not in sys.modules
assert FrozenTypeNet().encode(np.zeros((1,50,5)),np.array([50])).shape==(1,128)
assert summarize(np.ones((30,14)),np.ones((30,7))).shape==(42,)
print(json.dumps(FrozenFusion().predict_proba(d['face'],d['typing']).tolist()))
"""
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, check=True, text=True, capture_output=True)
    assert np.allclose(expected, json.loads(result.stdout), atol=2e-7)
