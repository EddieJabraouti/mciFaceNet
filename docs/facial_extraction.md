# Facial extraction from annotated video clips

The extractor emits the same 14 signal names and 42 summary-column names used
by the feature classifier. It is a **versioned reimplementation**, not a verified
reproduction of UFNet's unreleased extraction code. Existing fusion weights must
not be assumed to transfer to these features without a compatibility evaluation
or retraining.

## Run

Requires FFmpeg/ffprobe, Docker Desktop, and the extraction Python dependencies:

```sh
uv sync --extra train --extra extract
docker pull --platform linux/amd64 algebr/openface@sha256:f43ad4e7fa4530143c7a9e0e8eca7e4f2b45599c1ef19680b68ad1eebba05197
mkdir -p data/extractors/mediapipe
curl --fail --location \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
  --output data/extractors/mediapipe/face_landmarker.task
uv run --extra extract mcifacenet extract-video \
  --output data/processed/youtubepd_facial_v1 \
  --window-seconds 5 --stride-seconds 5 --workers 2
```

The MediaPipe model is SHA-256 verified before use. Docker reads the video folder
through a read-only mount and has networking disabled during extraction. On
macOS, MediaPipe's native runtime needs access to graphics services even with
CPU inference selected. The OpenFace image is the older Linux/amd64 image
linked by the project's official Docker instructions; its immutable digest is
recorded, rather than claiming it is the latest OpenFace release.

Use `--limit 1 --workers 1` with a separate output directory for a pilot.
Re-running the command reuses frame caches only after checking their input,
extractor configuration, source-code hashes and output hashes. Changing only
the window/stride recomputes summaries from those caches. Changing extraction
code, model, packages or geometry requires a new output directory. Re-running
with a limit rewrites that directory's summary tables for the limited selection.

## Outputs

- `frames/<clip_id>.npz`: presentation timestamps; 14 signals per frame; seven
  AU presence flags; 478 normalized 3D landmarks when exactly one face was found;
  tracking confidence/success, face count, box agreement, head pose and validity.
  Unavailable measurements use NaN; raw tracker predictions are retained even
  when rejected. Use the `valid` mask before aggregating or training. Load with
  `numpy.load(path, allow_pickle=False)`.
- `frames/<clip_id>.json`: input/configuration provenance and cache hashes.
- `openface/<clip_id>/openface.csv`: original OpenFace outputs, plus its log.
- `clip_features.csv`: one summary per extracted annotated clip, with labels,
  source links, timestamps, quality flags, AU sample counts and 42 named features.
- `window_features.csv`: the same summary schema for complete contiguous
  windows. Intervals are `[start,end)` in clip-relative seconds; add
  `source_start_seconds` for the original YouTube offset. Partial tails are not
  padded; short clips still have a clip-level summary.
- `protocol.json`, `progress.json`, `summary.json`: definitions, extraction
  outcomes, output counts and limitations.

Five seconds is an engineering starting setting for these mostly ten-second
clips, not a validated classification duration. No classifier is fitted or
invoked. Original source split annotations are metadata, not a guarantee of
participant-disjoint partitions. All clips/windows from the same real person
will need to remain together when defining a later training experiment.

## Signal definitions: `openface-static_mediapipe478_v1`

Seven AU intensities come directly from OpenFace: **01, 06, 12, 14, 25, 26, 45**.
Each has a separate binary presence predictor. Summaries use intensity values
only where the corresponding presence flag is one and the joint frame is valid.
Static AU mode (`-au_static`) avoids person-specific calibration over the entire
clip. It does **not** disable temporal smoothing: the pinned image's
`FaceAnalyser.cpp` applies a centered three-frame average to AU intensities and
a centered seven-frame majority vote to AU presence, including in static mode.
A window's boundary measurements can therefore depend on up to three neighboring
frames outside it. These are offline window summaries, not strictly causal or
independently extracted windows. Face trackers also retain earlier-frame state.
Keep all windows from a clip/person together in later dataset partitions. A
streaming implementation needs an explicit buffering or causal-extraction policy.
UFNet's precise AU mode is unverified, so this choice is recorded explicitly.

For MediaPipe, convert normalized coordinates to consistent units
`(x * image_width, y * image_height, z * image_width)`. Each measurement is the
3D Euclidean distance between the listed points, divided by the distance between
iris centers 468 and 473. These point choices are this implementation's explicit
definitions; UFNet's public paper does not specify its exact landmark indices.
Left/right mean anatomical left/right for an unmirrored image.

The pinned Face Landmarker model is a newer MediaPipe model; it is not verified
to reproduce UFNet's original Face Mesh coordinates.

| Signal | MediaPipe indices | Definition |
| --- | --- | --- |
| eye-open-right | 159, 145 | Right upper-to-lower eyelid |
| eye-open-left | 386, 374 | Left upper-to-lower eyelid |
| eye-raise-right | 105, 159 | Right eyebrow-to-upper-eyelid separation |
| eye-raise-left | 334, 386 | Left eyebrow-to-upper-eyelid separation |
| mouth-open | 13, 14 | Inner upper-to-lower lip separation |
| mouth-width | 61, 291 | Lip-corner separation |
| jaw-open | 13, 152 | Upper inner lip-to-chin separation |

`jaw-open` is a geometric proxy, not a measured mandibular joint angle. Head
pose is retained as auxiliary raw data and is not added to the 42-feature vector.
MediaPipe blendshape coefficients are not substituted for OpenFace AUs.

For each signal, export mean, population variance (`ddof=0`), and entropy:
`p_i = x_i / sum(x); H = -sum(p_i * ln(p_i))` over strictly positive terms.
Entropy is in nats and uses amplitude-normalized samples, not a histogram or
an entropy rate. Its magnitude depends on sample count: a constant positive
signal of N frames has entropy `ln(N)`. Matching window duration alone does not
eliminate differences in FPS, valid-frame count or active-AU count. Do not
interpret this statistic by itself as facial motion complexity.

An AU with no active frames has mean/variance/entropy zero, with its active count
recorded as zero. An entirely failed observation has missing features, not a
fabricated zero vector. UFNet's exact empty-set, entropy, variance, coordinate
and landmark conventions remain unverified. The retained `smile_` column prefix
is only a schema convention; these YouTube clips are not prompted-smiling tasks.

## Automatic quality checks

A jointly valid frame requires OpenFace success and confidence >=0.8, exactly
one MediaPipe face, box intersection-over-union >=0.3 between the two tools,
inter-iris distance >=10 pixels, finite nonnegative signals, AUs in [0,5] and
binary AU presence flags. Failed/multiple-face frames are retained and masked.

A summary passes automatic checks only with >=30 valid frames, >=80% valid-frame
coverage, and no multiple-face frames anywhere in its source clip. Other
summaries are marked `review_required`, with reasons. These thresholds are
engineering filters, not validated clinical thresholds. Review status is
independent of the recorded PD label. Face detections can miss people, and a
shot change can replace one face with another; these checks do not verify
patient identity or establish a clinically valid observation.

## References

- [UFNet code availability](https://github.com/ROC-HCI/UFNet#code)
- [Underlying feature method, Supplementary Note 2](https://arxiv.org/html/2308.02588v2)
- [OpenFace AU modes and predictions](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units)
- [Official OpenFace Docker instructions](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Docker)
- [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
