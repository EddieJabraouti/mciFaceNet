"""Joint attention over one facial token and ten frozen typing embeddings."""

import numpy as np
import torch
from torch import nn

from .fusion import FusionClassifier, FusionNetwork


GALLERY_SIZE = 10


class GalleryFusionNetwork(FusionNetwork):
    def __init__(self, width=16, heads=2, dropout=.35, input_dropout=.2,
                 modality_dropout=.15):
        super().__init__("attention", width, heads, dropout)
        if not 0 <= modality_dropout < 1:
            raise ValueError("Invalid modality dropout")
        self.input_dropout = nn.Dropout(input_dropout)
        self.modality_dropout = modality_dropout

    def forward(self, face, typing):
        face_token = self.face(self.input_dropout(face))[:, None, :]
        typing_tokens = self.typing(self.input_dropout(typing))
        if self.training and self.modality_dropout:
            draw = torch.rand(len(face), device=face.device)
            keep_face = draw >= self.modality_dropout / 2
            keep_typing = (draw < self.modality_dropout / 2) | (draw >= self.modality_dropout)
            face_token = face_token * keep_face[:, None, None]
            typing_tokens = typing_tokens * keep_typing[:, None, None]
        tokens = torch.cat([face_token, typing_tokens], dim=1)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + self.dropout(attended))
        # Pool AFTER joint attention. No chronology or gallery positions are invented.
        fused = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=1)
        return self.head(fused).squeeze(1)


class GalleryFusionClassifier(FusionClassifier):
    format_version = 3
    network_class = GalleryFusionNetwork

    def __init__(self, bundle):
        if bundle.get("gallery_size") != GALLERY_SIZE:
            raise ValueError("Expected a gallery of exactly ten embeddings")
        super().__init__(bundle)

    def prepare(self, face, typing):
        face, typing = np.asarray(face, dtype=np.float64), np.asarray(typing, dtype=np.float64)
        if (face.ndim != 2 or face.shape[1] != 42 or typing.shape != (len(face), GALLERY_SIZE, 128)
                or not np.isfinite(face).all() or not np.isfinite(typing).all()):
            raise ValueError("Expected finite face [N,42] and typing [N,10,128]")
        tensors = []
        for name, value in (("face", face), ("typing", typing)):
            mean, scale = self.scaling[name]
            normalized = ((value - mean) / scale).astype(np.float32)
            if not np.isfinite(normalized).all():
                raise ValueError("Features overflow preprocessing")
            tensors.append(torch.from_numpy(normalized))
        return tensors

    def predict_proba(self, face, typing, batch_size=256):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Positive batch size required")
        tensors = self.prepare(face, typing)
        output = []
        with torch.inference_mode():
            for start in range(0, len(tensors[0]), batch_size):
                batch = [t[start:start + batch_size] for t in tensors]
                p = torch.stack([model(*batch).sigmoid() for model in self.models]).mean(0).numpy()
                if not np.isfinite(p).all():
                    raise ValueError("Nonfinite gallery output")
                output.append(np.column_stack([1-p, p]))
        return np.concatenate(output) if output else np.empty((0, 2), dtype=np.float32)


def build_galleries(pairs, seed):
    """Fixed nonoverlapping sets; preserve partners, cycle their facial records."""
    required = ("typing_index", "typing_source_index", "typing_id", "typing_cohort",
                "face_index", "face_id", "face_source_row", "face_recording", "y")
    if any(k not in pairs for k in required):
        raise ValueError("Missing fixed-pair metadata")
    n = len(pairs["y"])
    if any(pairs[k].shape != (n,) for k in required):
        raise ValueError("Misaligned fixed-pair metadata")
    if len(np.unique(pairs["typing_source_index"])) != n:
        raise ValueError("Typing windows must be unique before grouping")
    for left, right in (("typing_id", "face_id"), ("face_id", "typing_id")):
        if any(len(set(pairs[right][pairs[left] == owner])) != 1 for owner in np.unique(pairs[left])):
            raise ValueError("Each participant must retain exactly one partner")
    rng, groups, face_rows, excluded = np.random.default_rng(seed), [], [], []
    for owner in np.unique(pairs["typing_id"]):
        rows = np.flatnonzero(pairs["typing_id"] == owner)
        if len(set(pairs["y"][rows])) != 1 or len(set(pairs["typing_cohort"][rows])) != 1:
            raise ValueError("Participant has inconsistent labels or cohort")
        rows = rows[np.argsort(pairs["typing_source_index"][rows])]
        rows = rng.permutation(rows)
        count = len(rows) // GALLERY_SIZE
        if not count:
            excluded.append({"typing_id": str(owner), "windows": len(rows),
                             "label": int(pairs["y"][rows[0]])})
        records = np.unique(pairs["face_index"][rows])
        records = rng.permutation(records)
        for g in range(count):
            groups.append(rows[g*GALLERY_SIZE:(g+1)*GALLERY_SIZE])
            face_rows.append(rows[np.flatnonzero(pairs["face_index"][rows] == records[g % len(records)])[0]])
    if not groups:
        raise ValueError("No complete ten-embedding galleries")
    groups, face_rows = np.asarray(groups), np.asarray(face_rows)
    result = {k: pairs[k][face_rows] for k in ("face_index", "face_id", "face_source_row", "face_recording")}
    result.update({k: pairs[k][groups] for k in ("typing_index", "typing_source_index")})
    result.update({k: pairs[k][groups[:, 0]] for k in ("typing_id", "typing_cohort", "y")})
    result["original_face_index"] = pairs["face_index"][groups]
    used = np.zeros(n, dtype=bool)
    used[groups.ravel()] = True
    owners = result["typing_id"]
    audit = {"galleries": len(groups), "typing_windows": int(used.sum()),
             "participants": len(np.unique(owners)),
             "positive_participants": len(np.unique(owners[result["y"] == 1])),
             "positive_galleries": int(result["y"].sum()),
             "facial_recordings": len(np.unique(result["face_index"])),
             "discarded_windows": int((~used).sum()), "excluded_participants": excluded,
             "windows_with_changed_facial_recording": int(np.sum(result["original_face_index"] != result["face_index"][:, None]))}
    return result, audit, pairs["typing_source_index"][~used]
