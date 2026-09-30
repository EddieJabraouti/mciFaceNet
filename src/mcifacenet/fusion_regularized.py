"""Regularized fusion with optional local attention over ordered keystrokes."""

import math

import numpy as np
import torch
from torch import nn

from .fusion import FusionClassifier, FusionNetwork


class RegularizedFusionNetwork(FusionNetwork):
    def __init__(self, width=16, heads=2, dropout=.35, input_dropout=.2,
                 modality_dropout=.15, sequence_radius=-1):
        super().__init__("attention", width, heads, dropout)
        if sequence_radius not in (-1, 4, 8, 50) or not 0 <= modality_dropout < 1:
            raise ValueError("Unsupported attention radius or modality dropout")
        self.sequence_radius, self.heads = sequence_radius, heads
        self.modality_dropout = modality_dropout
        self.input_dropout = nn.Dropout(input_dropout)
        if sequence_radius >= 0:
            self.key_projection = nn.Linear(5, width)
            self.key_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            self.key_norm = nn.LayerNorm(width)
            position = torch.arange(50).unsqueeze(1)
            frequencies = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.) / width))
            encoding = torch.zeros(50, width)
            encoding[:, 0::2], encoding[:, 1::2] = torch.sin(position * frequencies), torch.cos(position * frequencies)
            self.register_buffer("position_encoding", encoding)

    def attention_mask(self, lengths):
        """True blocks attention. Padded queries may see valid keys, then are discarded."""
        positions = torch.arange(50, device=lengths.device)
        valid = positions[None, :] < lengths[:, None]
        outside = (positions[:, None] - positions[None, :]).abs() > self.sequence_radius
        blocked = (outside[None, :, :] & valid[:, :, None]) | ~valid[:, None, :]
        return blocked.repeat_interleave(self.heads, dim=0), valid

    def sequence_tokens(self, sequence, lengths):
        mask, valid = self.attention_mask(lengths)
        sequence = sequence.masked_fill(~valid[:, :, None], 0.)
        tokens = self.key_projection(self.input_dropout(sequence)) + self.position_encoding
        attended, _ = self.key_attention(tokens, tokens, tokens, attn_mask=mask, need_weights=False)
        return self.key_norm(tokens + self.dropout(attended)).masked_fill(~valid[:, :, None], 0.)

    def forward(self, face, typing, sequence=None, lengths=None):
        face_token = self.face(self.input_dropout(face))
        typing_token = self.typing(self.input_dropout(typing))
        if self.sequence_radius >= 0:
            typing_token = typing_token + self.sequence_tokens(sequence, lengths).sum(1) / lengths[:, None]
        tokens = torch.stack([face_token, typing_token], dim=1)
        if self.training and self.modality_dropout:
            draw = torch.rand(len(tokens), device=tokens.device)
            keep = torch.stack([draw >= self.modality_dropout / 2,
                                (draw < self.modality_dropout / 2) | (draw >= self.modality_dropout)], dim=1)
            tokens = tokens * keep[:, :, None]
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + self.dropout(attended))
        return self.head(tokens.flatten(1)).squeeze(1)


class RegularizedFusionClassifier(FusionClassifier):
    format_version = 2
    network_class = RegularizedFusionNetwork

    def __init__(self, bundle):
        super().__init__(bundle)
        if self.bundle["architecture"]["sequence_radius"] >= 0:
            state = bundle["sequence_scaling"]
            mean, scale = np.asarray(state["mean"]), np.asarray(state["scale"])
            if (mean.shape != (5,) or scale.shape != (5,) or not np.isfinite(mean).all()
                    or not np.isfinite(scale).all() or (scale <= 0).any() or state.get("clip") != 5.):
                raise ValueError("Invalid sequence preprocessing")

    def prepare(self, face, typing, sequence=None, lengths=None):
        face, typing = np.asarray(face, dtype=np.float64), np.asarray(typing, dtype=np.float64)
        if (face.ndim != 2 or face.shape[1] != 42 or typing.shape != (len(face), 128)
                or not np.isfinite(face).all() or not np.isfinite(typing).all()):
            raise ValueError("Expected finite face [N,42] and typing [N,128]")
        result = []
        for name, value in (("face", face), ("typing", typing)):
            mean, scale = self.scaling[name]
            normalized = ((value - mean) / scale).astype(np.float32)
            if not np.isfinite(normalized).all():
                raise ValueError("Features overflow preprocessing")
            result.append(torch.from_numpy(normalized))
        if self.bundle["architecture"]["sequence_radius"] >= 0:
            sequence, lengths = np.asarray(sequence), np.asarray(lengths)
            if (sequence.shape != (len(face), 50, 5) or lengths.shape != (len(face),)
                    or not np.issubdtype(lengths.dtype, np.integer) or (lengths < 1).any()
                    or (lengths > 50).any() or not np.isfinite(sequence).all()):
                raise ValueError("Expected finite sequence [N,50,5] and integer lengths in [1,50]")
            state = self.bundle["sequence_scaling"]
            normalized = np.clip((sequence.astype(np.float64) - state["mean"]) / state["scale"], -5., 5.)
            normalized[np.arange(50)[None, :] >= lengths[:, None]] = 0.
            result.extend([torch.tensor(normalized, dtype=torch.float32), torch.tensor(lengths, dtype=torch.long)])
        return result

    def predict_proba(self, face, typing, sequence=None, lengths=None, batch_size=256):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Positive batch size required")
        tensors = self.prepare(face, typing, sequence, lengths)
        output = []
        with torch.inference_mode():
            for start in range(0, len(tensors[0]), batch_size):
                batch = [t[start:start + batch_size] for t in tensors]
                p = torch.stack([model(*batch).sigmoid() for model in self.models]).mean(0).numpy()
                if not np.isfinite(p).all():
                    raise ValueError("Nonfinite fusion output")
                output.append(np.column_stack([1-p, p]))
        return np.concatenate(output) if output else np.empty((0, 2), dtype=np.float32)
