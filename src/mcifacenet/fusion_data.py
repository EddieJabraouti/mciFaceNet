"""Verified frozen typing features and participant-aware artificial pairing."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROLES = ("train", "validation", "test")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Source checksum mismatch: {path}")
    return actual


def validate_typing(data, splits):
    required = {"x", "y", "owner", "split", "cohort", "synthetic"}
    if not required <= data.keys():
        raise ValueError("Typing cache is missing required arrays")
    x, y = data["x"], data["y"]
    if x.shape != (len(y), 128) or not np.isfinite(x).all():
        raise ValueError("Expected finite 128-coordinate typing embeddings")
    if any(data[k].shape != (len(y),) for k in required - {"x"}):
        raise ValueError("Typing metadata is not aligned with embeddings")
    if set(np.unique(y)) != {0, 1} or data["synthetic"].any():
        raise ValueError("Only real, explicitly binary-labeled typing windows are supported")
    if set(np.unique(data["split"])) != set(ROLES):
        raise ValueError("Expected train, validation, and test typing splits")
    owner_sets = {}
    for role in ROLES:
        owner_sets[role] = set(data["owner"][data["split"] == role])
        if owner_sets[role] != set(splits[role]):
            raise ValueError(f"Typing participant assignments changed: {role}")
        if set(data["y"][data["split"] == role]) != {0, 1}:
            raise ValueError(f"Both typing classes are required in {role}")
    if sum(map(len, owner_sets.values())) != len(set.union(*owner_sets.values())):
        raise ValueError("Typing participant leaks across splits")
    for owner in np.unique(data["owner"]):
        mask = data["owner"] == owner
        if not owner or len(set(y[mask])) != 1 or len(set(data["cohort"][mask])) != 1:
            raise ValueError("Typing participant has inconsistent metadata")


def load_typing(run):
    """Read exp4's registered cache; never regenerate or modify its data."""
    run = Path(run).resolve()
    root = run.parents[3]
    registration = json.loads((run / "registration.json").read_text())
    freeze = json.loads((run / "freeze.json").read_text())
    hashes = {"registration.json": checked(run / "registration.json", freeze["registration_sha256"]),
              "dataset.npz": checked(run / "dataset.npz", registration["dataset_sha256"])}
    if registration["features"] != 128 or registration["encoder_trainable_parameters"] != 0:
        raise ValueError("Expected exp4's frozen 128-coordinate representation")
    hashes[registration["encoder"]] = checked(root / registration["encoder"], registration["encoder_sha256"])
    with np.load(run / "dataset.npz", allow_pickle=False) as cache:
        data = {key: cache[key] for key in cache.files}
    validate_typing(data, registration["splits"])
    # Verify that cache columns/metadata still match their registered parent.
    for relative, digest in registration["cached_inputs"].items():
        path = root / relative
        hashes[relative] = checked(path, digest)
        with np.load(path, allow_pickle=False) as parent:
            if not np.array_equal(data["x"], parent["x"][:, :128]):
                raise ValueError("Typing embeddings differ from their source cache")
            for key in ("owner", "split", "y", "cohort", "synthetic"):
                if not np.array_equal(data[key], parent[key]):
                    raise ValueError(f"Typing source metadata changed: {key}")
    train_ids = set(registration["splits"]["train"])
    for fold in registration["folds"]:
        fit, score = set(fold["fit"]), set(fold["score"])
        if fit & score or fit | score != train_ids:
            raise ValueError("Existing typing classifier folds include held-out participants")
    groups = {}
    for role in ROLES:
        index = np.flatnonzero(data["split"] == role)
        groups[role] = {key: value[index] for key, value in data.items()}
        groups[role]["source_index"] = index
    audit = {"hashes": hashes, "encoder_sha256": registration["encoder_sha256"],
             "splits": {role: {"windows": len(d["y"]), "participants": len(set(d["owner"])),
                                "positive_participants": len(set(d["owner"][d["y"] == 1])),
                                "cohorts": {c: len(set(d["owner"][d["cohort"] == c]))
                                            for c in np.unique(d["cohort"])}}
                        for role, d in groups.items()},
             "representation": "Frozen exp4 TypeNet embedding of one 50-key sequence",
             "synthetic_typing_windows": 0,
             "source_label_mapping": {"0": "control", "1": "impaired"},
             "source_limitations": registration["limitations"]}
    return groups, audit


def load_typing_sequences(run, groups):
    """Recover ordered within-window features from the verified parent caches.

    No order across windows or recordings is inferred. Match every source row
    against the cached embedding and metadata before attaching its sequence.
    """
    run = Path(run).resolve()
    root = run.parents[3]
    registration = json.loads((run / "registration.json").read_text())
    parent_paths = list(registration["cached_inputs"])
    if len(parent_paths) != 1:
        raise ValueError("Expected one registered real-data parent")
    parent = root / parent_paths[0]
    checked(parent, registration["cached_inputs"][parent_paths[0]])
    parent_registration = parent.with_name("registration.json")
    parent_meta = json.loads(parent_registration.read_text())
    checked(parent, parent_meta["dataset_sha256"])
    pieces, hashes = [], {str(parent_registration.relative_to(root)): sha256(parent_registration)}
    # JSON insertion order is the original concatenation order; verify it below.
    for relative, digest in parent_meta["cached_inputs"].items():
        path = root / relative
        hashes[relative] = checked(path, digest)
        with np.load(path, allow_pickle=False) as cache:
            keep = ~cache["synthetic"] if "synthetic" in cache else np.ones(len(cache["y"]), dtype=bool)
            pieces.append({k: cache[k][keep] for k in ("sequences", "lengths", "owner", "y", "x")})
    combined = {k: np.concatenate([d[k] for d in pieces]) for k in pieces[0]}
    with np.load(run / "dataset.npz", allow_pickle=False) as cache:
        for key in ("owner", "y"):
            if not np.array_equal(combined[key], cache[key]):
                raise ValueError("Sequence source rows do not match registered typing rows")
        if not np.array_equal(combined["x"][:, :128], cache["x"]):
            raise ValueError("Sequence embeddings differ from registered cache")
    sequence, length = combined["sequences"], combined["lengths"]
    if (sequence.shape != (len(length), 50, 5) or not np.isfinite(sequence).all()
            or not np.issubdtype(length.dtype, np.integer) or (length < 1).any() or (length > 50).any()):
        raise ValueError("Invalid ordered typing windows")
    valid = np.arange(50)[None, :] < length[:, None]
    if np.any(sequence[~valid] != 0):
        raise ValueError("Expected zero-padded source sequences")
    result = {}
    for role, data in groups.items():
        index = data["source_index"]
        result[role] = {"sequence": sequence[index], "lengths": length[index]}
    return result, {"source_sha256": hashes, "full_windows": int((length == 50).sum()),
                    "partial_windows": int((length < 50).sum()),
                    "features": ["hold_latency", "inter_key_latency", "press_latency", "release_latency", "keycode_div_255"],
                    "ordering": "Original keystroke order inside each source window only; no cross-window chronology assumed"}


def face_arrays(rows):
    return {"x": np.asarray([r["features"] for r in rows], dtype=np.float64),
            "y": np.asarray([r["label"] for r in rows]),
            "owner": np.asarray([r["id"] for r in rows]),
            "recording": np.asarray([r["recording"] for r in rows]),
            "source_index": np.asarray([r["source_row"] for r in rows])}


def participant_weights(owners):
    _, inverse, counts = np.unique(owners, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.sum()


def fixed_participant_pairs(face, typing, seed):
    """One same-label face person per typist; each typing window appears once.

    Facial people whose recordings have conflicting labels are ineligible. All
    windows from a typist retain their assigned person and recording thereafter.
    """
    rng = np.random.default_rng(seed)
    face_rows = {owner: np.flatnonzero(face["owner"] == owner) for owner in np.unique(face["owner"])}
    typing_rows = {owner: np.flatnonzero(typing["owner"] == owner) for owner in np.unique(typing["owner"])}
    if any(len(set(typing["y"][rows])) != 1 for rows in typing_rows.values()):
        raise ValueError("Inconsistent typing participant labels")
    fi, ti = [], []
    for label in (0, 1):
        faces = [owner for owner, rows in face_rows.items() if set(face["y"][rows]) == {label}]
        typists = [owner for owner, rows in typing_rows.items() if set(typing["y"][rows]) == {label}]
        if not typists or len(faces) < len(typists):
            raise ValueError("Need enough distinct same-label facial participants for every typist")
        faces, typists = rng.permutation(faces), rng.permutation(typists)
        for face_owner, typing_owner in zip(faces, typists):
            windows = rng.permutation(typing_rows[typing_owner])
            recordings = rng.permutation(face_rows[face_owner])
            fi.extend(np.resize(recordings, len(windows)).tolist())
            ti.extend(windows.tolist())
    fi, ti = np.asarray(fi, dtype=np.int64), np.asarray(ti, dtype=np.int64)
    if not np.array_equal(np.sort(ti), np.arange(len(typing["y"]))) or not np.array_equal(face["y"][fi], typing["y"][ti]):
        raise ValueError("Fixed pairs must cover every typing window exactly once with matching labels")
    order = rng.permutation(len(ti))
    return fi[order], ti[order]


class PairSampler:
    """Sample a face, then a same-label typing person, then that person's window."""

    def __init__(self, face, typing):
        self.face, self.typing = face, typing
        self.face_weights = participant_weights(face["owner"])
        self.typing_rows = {owner: np.flatnonzero(typing["owner"] == owner)
                            for owner in np.unique(typing["owner"])}
        self.typing_people = {}
        for label in (0, 1):
            self.typing_people[label] = np.unique(typing["owner"][typing["y"] == label])
            if not len(self.typing_people[label]):
                raise ValueError("Both typing classes are required for pairing")
        if any(len(set(typing["y"][idx])) != 1 for idx in self.typing_rows.values()):
            raise ValueError("Inconsistent typing participant labels")

    def match(self, face_index, rng):
        typing_index = np.empty(len(face_index), dtype=np.int64)
        labels = self.face["y"][face_index]
        for label in (0, 1):
            positions = np.flatnonzero(labels == label)
            owners = rng.choice(self.typing_people[label], size=len(positions))
            for owner in np.unique(owners):
                selected = positions[owners == owner]
                typing_index[selected] = rng.choice(self.typing_rows[owner], size=len(selected))
        if not np.array_equal(labels, self.typing["y"][typing_index]):
            raise AssertionError("Pairing changed the label")
        return np.asarray(face_index), typing_index

    def training(self, count, rng):
        face_index = rng.choice(len(self.face["y"]), size=count, p=self.face_weights)
        return self.match(face_index, rng)

    def evaluation(self, draws, seed):
        pairs = [self.match(np.arange(len(self.face["y"])), np.random.default_rng(seed + draw))
                 for draw in range(draws)]
        return (np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs]),
                np.repeat(np.arange(draws), len(self.face["y"])))
