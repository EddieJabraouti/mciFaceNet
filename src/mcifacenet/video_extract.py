"""Local OpenFace/MediaPipe extraction with auditable frame caches and summaries."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .facial_signals import (AUS, EXTRACTOR_VERSION, GEOMETRY_PAIRS, SIGNALS,
                             bbox_iou, geometry, summarize, windows)
from .model import FEATURES

OPENFACE_IMAGE = "algebr/openface@sha256:f43ad4e7fa4530143c7a9e0e8eca7e4f2b45599c1ef19680b68ad1eebba05197"
OPENFACE_BIN = "/home/openface-build/build/bin/FeatureExtraction"
MEDIAPIPE_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
DEFAULT_MANIFEST = Path("data/upstream/youtubepd/download_manifest.json")
DEFAULT_MODEL = Path("data/extractors/mediapipe/face_landmarker.task")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def frame_times(video):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
         "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    times = np.array([float(f["best_effort_timestamp_time"])
                      for f in json.loads(result.stdout)["frames"]])
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Video needs finite, strictly increasing presentation timestamps")
    return times - times[0]


def run_openface(video, target):
    target.mkdir(parents=True, exist_ok=True)
    name = f"mcifacenet-{os.getpid()}-{video.stem}".lower().replace("_", "-")
    command = [
        "docker", "run", "--rm", "--name", name, "--network", "none",
        "--platform", "linux/amd64", "--cpus", "2", "--memory", "4g",
        "-e", "OMP_NUM_THREADS=2", "-e", "OPENBLAS_NUM_THREADS=1",
        "--mount", f"type=bind,src={video.parent},dst=/input,readonly",
        "--mount", f"type=bind,src={target},dst=/output",
        "--workdir", str(Path(OPENFACE_BIN).parent), "--entrypoint", OPENFACE_BIN,
        OPENFACE_IMAGE, "-f", f"/input/{video.name}", "-out_dir", "/output",
        "-of", "openface.csv", "-aus", "-2Dfp", "-pose", "-au_static",
    ]
    with (target / "extraction.log").open("w") as log:
        try:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
            raise
    path = target / "openface.csv"
    if not path.exists():
        raise ValueError(f"OpenFace did not create {path}; see extraction.log")
    return path


def read_openface(path, timestamps):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]
    required = {"frame", "timestamp", "confidence", "success"}
    required |= {f"{au}_{suffix}" for au in AUS for suffix in ("r", "c")}
    required |= {f"{axis}_{i}" for axis in ("x", "y") for i in range(68)}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("OpenFace output is missing required AU, tracking, or landmark columns")
    n = len(timestamps)
    intensity, presence = np.full((n, 7), np.nan), np.full((n, 7), np.nan)
    confidence, success = np.zeros(n), np.zeros(n, dtype=bool)
    boxes, pose = np.full((n, 4), np.nan), np.full((n, 3), np.nan)
    seen = set()
    tolerance = max(0.1, float(np.median(np.diff(timestamps))) * 1.5) if n > 1 else 0.1
    for row in rows:
        number = int(row["frame"])
        if not 1 <= number <= n or number in seen:
            raise ValueError("OpenFace frame numbers are duplicated or outside the decoded video")
        seen.add(number)
        i = number - 1
        timestamp = float(row["timestamp"])
        if not np.isfinite(timestamp) or abs(timestamp - timestamps[i]) > tolerance:
            raise ValueError("OpenFace timestamps disagree with video frame alignment")
        intensity[i] = [float(row[f"{au}_r"]) for au in AUS]
        presence[i] = [float(row[f"{au}_c"]) for au in AUS]
        confidence[i] = float(row["confidence"])
        success[i] = int(row["success"]) == 1
        x, y = ([float(row[f"{axis}_{j}"]) for j in range(68)] for axis in ("x", "y"))
        boxes[i] = [min(x), min(y), max(x), max(y)]
        pose[i] = [float(row.get(f"pose_R{axis}", "nan")) for axis in ("x", "y", "z")]
    return dict(au_intensity=intensity, au_presence=presence, openface_confidence=confidence,
                openface_success=success, openface_bbox=boxes, head_pose_radians=pose)


def extract_frames(video, model, timestamps, openface):
    import cv2
    import mediapipe as mp

    cv2.setNumThreads(1)
    n = len(timestamps)
    landmarks = np.full((n, 478, 3), np.nan, dtype=np.float32)
    geometry_values = np.full((n, 7), np.nan)
    face_count = np.zeros(n, dtype=np.int16)
    iris_distance, overlap = np.full(n, np.nan), np.zeros(n)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=3,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    capture = cv2.VideoCapture(str(video))
    previous_ms = -1
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            for i, timestamp in enumerate(timestamps):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Decoder stopped at frame {i}, expected {n}")
                height, width = frame.shape[:2]
                timestamp_ms = max(previous_ms + 1, round(float(timestamp) * 1000))
                previous_ms = timestamp_ms
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                    timestamp_ms,
                )
                face_count[i] = len(result.face_landmarks)
                # Do not silently combine OpenFace's person with a different MediaPipe face.
                if face_count[i] != 1:
                    continue
                points = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]])
                landmarks[i] = points
                xy = points[:468, :2] * [width, height]
                box = np.r_[xy.min(axis=0), xy.max(axis=0)]
                overlap[i] = bbox_iou(box, openface["openface_bbox"][i])
                try:
                    geometry_values[i], iris_distance[i] = geometry(points, width, height)
                except ValueError:
                    continue
            if capture.read()[0]:
                raise ValueError("Video decoder returned more frames than ffprobe")
    finally:
        capture.release()
    signals = np.column_stack((openface["au_intensity"], geometry_values))
    valid = (openface["openface_success"] & (openface["openface_confidence"] >= 0.8)
             & (face_count == 1) & (overlap >= 0.3) & np.isfinite(signals).all(axis=1)
             & (signals >= 0).all(axis=1) & (signals[:, :7] <= 5).all(axis=1)
             & np.isin(openface["au_presence"], [0, 1]).all(axis=1))
    return {"timestamp_seconds": timestamps, "signals": signals, "valid": valid,
            "face_count": face_count, "landmarks": landmarks, "iris_distance_pixels": iris_distance,
            "face_bbox_iou": overlap, "image_size": np.array([width, height]), **openface}


def extract_one(record, root, output, model, protocol):
    clip_id = record["clip_id"]
    video = (root / record["mp4_path"]).resolve()
    if sha256(video) != record["verification"]["sha256"]:
        raise ValueError(f"Source MP4 checksum changed: {clip_id}")
    signature = {"video_sha256": record["verification"]["sha256"], "protocol": protocol}
    cache_path, metadata_path = output / "frames" / f"{clip_id}.npz", output / "frames" / f"{clip_id}.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["signature"] != signature:
            raise ValueError(f"Cached extractor settings differ for {clip_id}; use a new output directory")
        if sha256(cache_path) != metadata["frames_sha256"]:
            raise ValueError(f"Frame cache checksum mismatch for {clip_id}")
        if sha256(output / "openface" / clip_id / "openface.csv") != metadata["openface_csv_sha256"]:
            raise ValueError(f"OpenFace cache checksum mismatch for {clip_id}")
        return metadata
    timestamps = frame_times(video)
    if len(timestamps) != record["verification"]["frames"]:
        raise ValueError(f"Frame count differs from verified download: {clip_id}")
    raw_csv = run_openface(video, output / "openface" / clip_id)
    openface = read_openface(raw_csv, timestamps)
    frames = extract_frames(video, model, timestamps, openface)
    temporary = cache_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **frames)
    temporary.replace(cache_path)
    metadata = {"clip_id": clip_id, "signature": signature, "frames_sha256": sha256(cache_path),
                "openface_csv_sha256": sha256(raw_csv), "frames": len(timestamps),
                "valid_frames": int(frames["valid"].sum()),
                "multiple_face_frames": int((frames["face_count"] > 1).sum()),
                "completed_at": datetime.now(timezone.utc).isoformat()}
    write_json(metadata_path, metadata)
    return metadata


def write_table(path, rows):
    if not rows:
        fields = ["clip_id", "quality_status", *FEATURES]
    else:
        fields = list(rows[0])
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def extract_videos(manifest_path, output, model_path=DEFAULT_MODEL, window_seconds=5.0,
                   stride_seconds=5.0, workers=2, limit=None):
    manifest_path, output, model = Path(manifest_path).resolve(), Path(output).resolve(), Path(model_path).resolve()
    # Resolve manifest paths from the repository root, not the caller's cwd.
    root = manifest_path.parents[3]
    manifest = json.loads(manifest_path.read_text())
    records = [r for r in manifest["records"] if r["status"] == "downloaded"]
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive")
        records = records[:limit]
    if not records or workers not in (1, 2, 3, 4):
        raise ValueError("Need downloaded records and 1–4 workers")
    windows(1.0, window_seconds, stride_seconds)  # Validate before expensive extraction.
    if sha256(model) != MEDIAPIPE_SHA256:
        raise ValueError("MediaPipe model does not match the pinned model checksum")
    if any(not (output / "frames" / f"{r['clip_id']}.json").exists() for r in records):
        try:
            subprocess.run(["docker", "image", "inspect", OPENFACE_IMAGE], check=True,
                           capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as error:
            raise ValueError("OpenFace is unavailable: start Docker and pull the pinned image; "
                             "see docs/facial_extraction.md") from error
    versions = {package: importlib.metadata.version(package)
                for package in ("mediapipe", "opencv-contrib-python", "numpy")}
    protocol = {
        "extractor_version": EXTRACTOR_VERSION, "openface_image": OPENFACE_IMAGE,
        "openface_au_mode": "static", "mediapipe_sha256": MEDIAPIPE_SHA256,
        "package_versions": versions, "signals": list(SIGNALS),
        "geometry_pairs": {k: list(v) for k, v in GEOMETRY_PAIRS.items()},
        "geometry_coordinates": "3D Euclidean: x*width,y*height,z*width; normalize by distance(468,473)",
        "openface_confidence_min": 0.8, "mediapipe_max_faces": 3, "bbox_iou_min": 0.3,
        "iris_pixels_min": 10, "au_active_frames_only": True,
        "variance_ddof": 0, "entropy": "-sum(p_i*ln(p_i)), p_i=x_i/sum(x); zero for empty/all-zero",
        "compatibility": "UFNet feature names only; exact numerical/weight compatibility is unverified",
        "aggregation_source_sha256": sha256(Path(__file__).with_name("facial_signals.py")),
        "extraction_source_sha256": sha256(__file__),
    }
    for subdir in ("frames", "openface"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Output directory contains a different extraction protocol")
    write_json(protocol_path, protocol)
    statuses = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_one, r, root, output, model, protocol): r for r in records}
        for done, future in enumerate(as_completed(futures), 1):
            record = futures[future]
            try:
                metadata = future.result()
                statuses[record["clip_id"]] = {"status": "extracted", **metadata}
                print(f"[{done}/{len(records)}] {record['clip_id']}: {metadata['valid_frames']}/{metadata['frames']} valid frames", flush=True)
            except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
                statuses[record["clip_id"]] = {"status": "failed", "error": str(error)}
                print(f"[{done}/{len(records)}] {record['clip_id']}: failed: {error}", flush=True)
            write_json(output / "progress.json", statuses)
    clip_rows, window_rows = [], []
    for record in records:
        clip_id = record["clip_id"]
        if statuses[clip_id]["status"] != "extracted":
            continue
        with np.load(output / "frames" / f"{clip_id}.npz", allow_pickle=False) as data:
            frames = dict(data)
        base = {"clip_id": clip_id, "source_video_url": record["url"], "source_excel_row": record["excel_row"],
                "source_file": record["source_file"], "pd": record["pd"], "label": int(record["pd"] == "y"),
                "source_split": record["split_raw"], "source_start_seconds": record["start_seconds"],
                "extractor_version": EXTRACTOR_VERSION, "ufnet_numerical_compatibility": "unverified"}
        multi = bool(np.any(frames["face_count"] > 1))
        duration = record["expected_duration_seconds"]
        clip_rows.append({**base, **summarize(frames, 0.0, float(duration), multi)})
        for start, end in windows(duration, window_seconds, stride_seconds):
            window_rows.append({**base, **summarize(frames, start, end, multi)})
    write_table(output / "clip_features.csv", clip_rows)
    write_table(output / "window_features.csv", window_rows)
    summary = {
        "clips_requested": len(records), "clips_extracted": len(clip_rows),
        "clips_failed": len(records) - len(clip_rows),
        "clip_summaries_passing_checks": sum(r["quality_status"] == "passes_automatic_checks" for r in clip_rows),
        "window_seconds": window_seconds, "stride_seconds": stride_seconds,
        "window_summaries": len(window_rows),
        "window_summaries_passing_checks": sum(r["quality_status"] == "passes_automatic_checks" for r in window_rows),
        "source_manifest_sha256": sha256(manifest_path), "feature_columns": list(FEATURES),
        "frame_signals": list(SIGNALS), "output": str(output),
        "limitations": ["Exact UFNet numerical compatibility unverified; do not assume old fusion weights transfer.",
                        "Automatic face checks do not establish participant identity; clips may contain scene cuts.",
                        "Multiple-face clips require target review; no identity is inferred from source clip labels.",
                        "Five-second windows and quality thresholds are engineering defaults, not validated clinical settings.",
                        "Entropy depends on frame count; comparisons require matching duration and sampling policy."],
    }
    write_json(output / "summary.json", summary)
    return summary
