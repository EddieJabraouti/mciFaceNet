"""Download the annotated YouTubePD intervals, retaining original source metadata.

Adapts the yt-dlp + FFmpeg time-crop step of samwli/YouTubePD-data's
prepare_data.py at 43797386a65ffb58db53628b90ef8e8f35512e0d. The unmodified
upstream script is cached beside the source spreadsheets. Face cropping,
resizing, frame-rate conversion, feature extraction and training are excluded.

Requires yt-dlp[default], openpyxl, FFmpeg/ffprobe and Node.js. Run from the repo:
    .venv/bin/python scripts/prepare_youtubepd.py --prepare-only
    .venv/bin/python scripts/prepare_youtubepd.py

Completed files are verified and skipped on restart; failed downloads are retried.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/upstream/youtubepd"
COMMIT = "43797386a65ffb58db53628b90ef8e8f35512e0d"
SOURCE = DEST / COMMIT


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def timestamp_seconds(value):
    # The primary workbook displays h:mm, but its entries mean minutes:seconds.
    # E.g. Excel time(8,18) is video offset 08:18 (498s), not 8 hours 18 minutes.
    # The supplementary sheet uses explicit M:SS strings. Never guess typos.
    if isinstance(value, time):
        if value.second or value.microsecond or value.minute >= 60:
            raise ValueError(f"Unexpected Excel time: {value}")
        return value.hour * 60 + value.minute
    text = str(value).strip()
    if not re.fullmatch(r"\d+:\d{2}", text):
        raise ValueError(f"Malformed M:SS timestamp: {text!r}")
    minutes, seconds = map(int, text.split(":"))
    if seconds >= 60:
        raise ValueError(f"Seconds outside 0..59: {text!r}")
    return minutes * 60 + seconds


def source_records():
    records, hashes = [], {}
    for filename in ("data_sheet.xlsx", "NegSamples.xlsx"):
        path = SOURCE / "data_sheets" / filename
        hashes[str(path.relative_to(ROOT))] = sha256(path)
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheet = workbook.active
        for row_number, row in enumerate(sheet.values, 1):
            if not row[0] or str(row[0]).strip() == "link":
                continue
            clip_id = f"{path.stem}_row{row_number:04d}"
            url = str(row[0]).strip()
            record = {
                "clip_id": clip_id, "source_file": str(path.relative_to(ROOT)),
                "sheet": sheet.title, "excel_row": row_number, "url": url,
                "start_raw": str(row[3]), "end_raw": str(row[4]),
                "timestamp_policy": "Excel h:mm interpreted as M:SS; strings as M:SS",
                "pd_raw": row[2], "pd": "n" if filename == "NegSamples.xlsx" else str(row[2]).strip(),
                "label_provenance": "negative-only source sheet" if filename == "NegSamples.xlsx" else "parkinson y/n column",
                "year_raw": row[1], "notes_raw": row[5], "split_raw": row[6],
                "severity_raw": row[7], "confidence_raw": row[8],
                "mp4_path": str((DEST / "mp4" / f"{clip_id}.mp4").relative_to(ROOT)),
                "status": "pending", "attempts": 0,
            }
            try:
                if not re.fullmatch(r"https://(?:www\.)?(?:youtube\.com/watch\?[^\s]+|youtu\.be/[^\s]+)", url):
                    raise ValueError("Source link is not a supported YouTube URL")
                record["start_seconds"] = timestamp_seconds(row[3])
                record["end_seconds"] = timestamp_seconds(row[4])
                record["expected_duration_seconds"] = record["end_seconds"] - record["start_seconds"]
                if record["expected_duration_seconds"] <= 0:
                    raise ValueError("End timestamp must be later than start timestamp")
            except ValueError as error:
                record.update(status="invalid_source", error=str(error))
            records.append(record)
        workbook.close()
    return records, hashes


def verify_video(path, expected_duration):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    probe = json.loads(result.stdout)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    duration = float(video.get("duration", probe["format"]["duration"]))
    if abs(duration - expected_duration) > 0.25:
        raise ValueError(f"Video duration {duration:.3f}s differs from requested {expected_duration}s")
    if int(video.get("nb_frames", 0)) < 1 or not video["width"] or not video["height"]:
        raise ValueError("No video frames in downloaded file")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True, text=True, timeout=120, check=True,
    )
    return {"duration_seconds": duration, "width": video["width"], "height": video["height"],
            "frame_rate": video["avg_frame_rate"], "frames": int(video["nb_frames"]),
            "codec": video["codec_name"], "bytes": path.stat().st_size, "sha256": sha256(path)}


def download(record):
    record = dict(record)
    output = ROOT / record["mp4_path"]
    if record["status"] == "downloaded" and output.exists():
        verified = verify_video(output, record["expected_duration_seconds"])
        if verified["sha256"] != record["verification"]["sha256"]:
            raise ValueError(f"Previously downloaded file changed: {output}")
        return record
    record["attempts"] += 1
    log_path = DEST / "logs" / f"{record['clip_id']}.attempt{record['attempts']}.log"
    record["log_path"] = str(log_path.relative_to(ROOT))
    # Stage outputs away from mp4/ so failed/partial files never look complete.
    staging = DEST / "partial" / record["clip_id"]
    staging.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist", "--no-progress",
        "--js-runtimes", f"node:{shutil.which('node')}", "--socket-timeout", "20",
        "--retries", "2", "--extractor-retries", "2", "--fragment-retries", "2",
        "--no-mtime", "--force-overwrites", "--download-sections",
        f"*{record['start_seconds']}-{record['end_seconds']}", "--force-keyframes-at-cuts",
        "--merge-output-format", "mp4", "--recode-video", "mp4", "-f",
        "bv*[ext=mp4][vcodec^=avc1]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "-o", str(staging / "clip.%(ext)s"), record["url"],
    ]
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError("Download exceeded 300 seconds")
        if code:
            errors = [line for line in log_path.read_text(errors="replace").splitlines() if "ERROR:" in line]
            raise RuntimeError(errors[-1] if errors else f"yt-dlp exited with status {code}; see log")
        staged_file = staging / "clip.mp4"
        record["verification"] = verify_video(staged_file, record["expected_duration_seconds"])
        if output.exists():
            raise FileExistsError(f"Refusing to replace unregistered existing MP4: {output}")
        staged_file.rename(output)
        record.update(status="downloaded", completed_at=datetime.now(timezone.utc).isoformat())
        record.pop("error", None)
    except (OSError, RuntimeError, ValueError, StopIteration, subprocess.SubprocessError) as error:
        record.update(status="failed", error=str(error))
    return record


def save_manifest(manifest):
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["counts"] = dict(Counter(r["status"] for r in manifest["records"]))
    tmp = DEST / "download_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    tmp.replace(DEST / "download_manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2, 3))
    args = parser.parse_args()
    for name in ("mp4", "logs", "partial"):
        (DEST / name).mkdir(parents=True, exist_ok=True)
    records, hashes = source_records()
    manifest_path = DEST / "download_manifest.json"
    manifest = {
        "upstream_commit": COMMIT,
        "upstream_script_url": f"https://github.com/samwli/YouTubePD-data/blob/{COMMIT}/prepare_data.py",
        "upstream_script_sha256": sha256(SOURCE / "prepare_data.py"),
        "source_sha256": hashes, "adapted_script_sha256": sha256(__file__),
        "processing": "Annotated time intervals only; native selected-stream resolution/frame rate; H.264 preference; exact cuts re-encoded by FFmpeg; audio retained when available.",
        "records": records,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["source_sha256"] != hashes:
            raise ValueError("Source spreadsheets changed since the existing download manifest")
        manifest["records"] = previous["records"]
    save_manifest(manifest)
    print(json.dumps(manifest["counts"]), flush=True)
    if args.prepare_only:
        return
    for binary in ("ffmpeg", "ffprobe", "node"):
        if not shutil.which(binary):
            raise RuntimeError(f"Required executable not found: {binary}")
    manifest["yt_dlp_version"] = subprocess.check_output([sys.executable, "-m", "yt_dlp", "--version"], text=True).strip()
    eligible = [(i, r) for i, r in enumerate(manifest["records"]) if r["status"] != "invalid_source"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, r): i for i, r in eligible}
        for number, future in enumerate(as_completed(futures), 1):
            record = future.result()
            manifest["records"][futures[future]] = record
            save_manifest(manifest)
            detail = record.get("error", "verified")
            print(f"[{number}/{len(eligible)}] {record['clip_id']}: {record['status']} {detail}", flush=True)
    print(json.dumps(manifest["counts"]), flush=True)


if __name__ == "__main__":
    main()
