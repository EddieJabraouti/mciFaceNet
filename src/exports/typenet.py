"""Frozen TypeNet encoder and the clinical typing adapter's numeric input policy."""

import numpy as np
import torch

from ._integrity import checkpoint
from ._typenet import TypeNetEncoder, events_to_features


class FrozenTypeNet:
    def __init__(self):
        self._encoder = TypeNetEncoder()
        self._encoder.load_state_dict(torch.load(checkpoint("typenet"), map_location="cpu", weights_only=True))
        self._encoder.eval().requires_grad_(False)

    def prepare_events(self, events):
        """Paired events: {key, press_ms, release_ms}; returns [S,50,5], [S].

        Only printable ASCII is retained. No text is written to disk. Duplicate
        events are removed; invalid timing raises rather than inventing a value.
        Call once per recording/session, not across unrelated sessions.
        """
        neutral = float(self._encoder.input_batch_norm.running_mean[4]) * 255.
        ordered, seen = [], set()
        for i, event in enumerate(events):
            key = event["key"]
            key = " " if key == "spacebar" else key
            if not isinstance(key, str) or not (len(key) == 1 and key.isascii() and key.isprintable()):
                continue
            press, release = float(event["press_ms"]), float(event["release_ms"])
            if not np.isfinite([press, release]).all() or not release > press >= 0:
                raise ValueError("Expected finite release_ms > press_ms >= 0")
            identity = (key, press, release)
            if identity in seen:
                continue
            seen.add(identity)
            code = ord(key.upper()) if key.isalnum() else 32 if key == " " else neutral
            ordered.append((press, release, code, i))
        ordered.sort(key=lambda e: (e[0], e[3]))
        windows = [events_to_features(ordered[start:start+51], 50) for start in range(0, len(ordered), 50)]
        if not windows:
            return np.empty((0, 50, 5), dtype=np.float32), np.empty(0, dtype=np.int64)
        return np.stack([w[0] for w in windows]), np.array([w[1] for w in windows], dtype=np.int64)

    def encode(self, sequence, lengths, batch_size=256):
        x, lengths = np.asarray(sequence, dtype=np.float32), np.asarray(lengths)
        if (x.ndim != 3 or x.shape[1:] != (50, 5) or lengths.shape != (len(x),)
                or not np.issubdtype(lengths.dtype, np.integer) or (lengths < 1).any() or (lengths > 50).any()
                or not np.isfinite(x).all() or not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("Expected finite sequence[N,50,5], integer lengths[N] in 1..50 and positive batch size")
        if not len(x):
            return np.empty((0, 128), dtype=np.float32)
        self._encoder.eval()
        output = []
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                output.append(self._encoder(torch.from_numpy(x[start:start+batch_size]),
                                            torch.tensor(lengths[start:start+batch_size], dtype=torch.long)).numpy())
        result = np.concatenate(output)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite TypeNet output")
        return result
