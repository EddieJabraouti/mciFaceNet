"""Attention/concatenation fusion of facial summaries and frozen typing embeddings."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import FEATURES


class FusionNetwork(nn.Module):
    def __init__(self, kind="attention", width=32, heads=4, dropout=0.1):
        super().__init__()
        if kind not in ("attention", "concatenation"):
            raise ValueError("Unknown fusion architecture")
        self.kind = kind
        self.face = nn.Sequential(nn.Linear(42, width), nn.GELU(), nn.LayerNorm(width))
        self.typing = nn.Sequential(nn.Linear(128, width), nn.GELU(), nn.LayerNorm(width))
        if kind == "attention":
            self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(width)
            self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, face, typing):
        tokens = torch.stack([self.face(face), self.typing(typing)], dim=1)
        if self.kind == "attention":
            attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
            tokens = self.norm(tokens + self.dropout(attended))
        return self.head(tokens.flatten(1)).squeeze(1)


class FusionClassifier:
    """Feature-input inference: labels and participant matching are never inputs."""

    format_version = 1
    network_class = FusionNetwork

    def __init__(self, bundle):
        if (bundle.get("format_version") != self.format_version or bundle.get("classes") != ["control", "impaired"]
                or bundle.get("face_features") != list(FEATURES) or bundle.get("typing_dimensions") != 128
                or bundle.get("threshold") != 0.5):
            raise ValueError("Unsupported fusion artifact schema")
        self.bundle = bundle
        self.scaling = {}
        for name, width in (("face", 42), ("typing", 128)):
            mean = np.asarray(bundle["scaling"][name]["mean"], dtype=np.float64)
            scale = np.asarray(bundle["scaling"][name]["scale"], dtype=np.float64)
            if (mean.shape != (width,) or scale.shape != (width,) or not np.isfinite(mean).all()
                    or not np.isfinite(scale).all() or (scale <= 0).any()):
                raise ValueError("Invalid fusion preprocessing")
            self.scaling[name] = (mean, scale)
        if not bundle["states"]:
            raise ValueError("Fusion ensemble has no models")
        self.models = []
        for state in bundle["states"]:
            if not all(torch.isfinite(value).all() for value in state.values()):
                raise ValueError("Nonfinite fusion weights")
            model = self.network_class(**bundle["architecture"])
            model.load_state_dict(state, strict=True)
            model.eval().requires_grad_(False)
            self.models.append(model)

    @classmethod
    def load(cls, path):
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    def export(self, path):
        with Path(path).open("xb") as stream:
            torch.save(self.bundle, stream)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def predict_proba(self, facial, typing, batch_size=512):
        facial, typing = np.asarray(facial, dtype=np.float64), np.asarray(typing, dtype=np.float64)
        if (facial.ndim != 2 or facial.shape[1] != 42 or typing.shape != (len(facial), 128)
                or not np.isfinite(facial).all() or not np.isfinite(typing).all()
                or not isinstance(batch_size, int) or batch_size <= 0):
            raise ValueError("Expected aligned finite arrays: facial [N,42], typing [N,128]")
        tensors = []
        for name, x in (("face", facial), ("typing", typing)):
            mean, scale = self.scaling[name]
            normalized = ((x - mean) / scale).astype(np.float32)
            if not np.isfinite(normalized).all():
                raise ValueError("Features overflow fusion preprocessing")
            tensors.append(torch.from_numpy(normalized))
        output = []
        with torch.inference_mode():
            for start in range(0, len(facial), batch_size):
                f, t = [x[start:start + batch_size] for x in tensors]
                p = torch.stack([model(f, t).sigmoid() for model in self.models]).mean(0).numpy()
                if not np.isfinite(p).all():
                    raise ValueError("Nonfinite fusion output")
                output.append(np.column_stack([1 - p, p]))
        return np.concatenate(output) if output else np.empty((0, 2), dtype=np.float32)
