# Frozen numeric-input modules

Three separate modules, linked by `manifest.json`. Copy this entire `exports`
directory into a Python import path. Runtime dependencies are **NumPy and
PyTorch**; facial aggregation alone needs only NumPy. Tested versions are in
the manifest. The package does not import the training repository or require
OpenFace, MediaPipe, Docker, a camera, or network access.

The frontend computes facial measurements locally and sends numerical arrays.
These modules have no image/video input, do not open a camera, and do not save
user inputs. Frame timestamps and quality metadata should accompany collection
outside the model to define reproducible observation windows.

| Module | Entry point | Input | Output |
|---|---|---|---|
| TypeNet encoder | `exports.typenet.FrozenTypeNet` | `sequence[N,50,5]`, `lengths[N]` | `embedding[N,128]` |
| Facial summary | `exports.facial.summarize` | usable `measurements[T,14]`, `au_presence[T,7]` | `face[42]` |
| Gallery attention fusion | `exports.fusion.FrozenFusion` | `face[N,42]`, `typing[N,10,128]` | probabilities `[N,2]` |

`N` is an inference batch size, not a participant identifier. No labels or
participant IDs are model inputs. Collection must ensure each gallery has ten
distinct sequences from the same user and the facial summary belongs to that
user. The development experiment used synthetic same-label participant pairs.

## 1. TypeNet encoder

`weights/typenet.pt` is the unchanged frozen encoder checkpoint (200,458
parameters). `FrozenTypeNet()` verifies its checksum, loads on CPU, disables
gradients, and uses evaluation mode. It returns unnormalized 128-dimensional
embeddings, one per sequence. There is no gallery averaging in this module.

For each key `i`, the five input columns are:

1. **HL:** `release_i - press_i`, in seconds.
2. **IL:** `press_(i+1) - release_i`, in seconds. Negative values from overlapping
   keystrokes are retained.
3. **PL:** `press_(i+1) - press_i`, in seconds.
4. **RL:** `release_(i+1) - release_i`, in seconds.
5. **Mapped key code / 255**.

Each sequence has up to 50 keys. Supply its actual integer length in `1..50`
and zero-pad unused rows. The encoder masks padding. A following 51st event may
be used to compute the 50th key's IL/PL/RL; it does not become a 51st embedding
input. At a recording's end, unavailable next-key latencies are zero.

The optional `prepare_events(events)` helper accepts a session's already paired
events: `{"key": "a", "press_ms": 1000, "release_ms": 1080}`. It returns
sequences and lengths. It sorts by press time, removes exact duplicate events,
and retains printable ASCII keys. ASCII letters map to uppercase codes, digits
keep their codes, space maps to 32, and other printable punctuation maps to the
checkpoint's neutral key value. Nonprinting/unmapped keys are excluded; invalid
timing raises an error. Call separately for each recording/session.

Pass raw timing features to `encode`; the encoder applies its saved batch-norm
statistics. Do not add L2 normalization or apply the fusion scaler here.

```python
from exports.typenet import FrozenTypeNet

encoder = FrozenTypeNet()
sequences, lengths = encoder.prepare_events(paired_key_events)
embeddings = encoder.encode(sequences, lengths)  # [S,128]
```

## 2. Facial measurement aggregation

This is a deterministic numerical module with **no learned weights**. The
frontend owns face tracking, AU estimation, geometric measurements, and frame
quality checks. The exported module summarizes already usable frames.

The input measurement columns, in exact order, are:

| Index | Measurement | Definition |
|---:|---|---|
| 0 | AU01 | Inner brow raiser intensity |
| 1 | AU06 | Cheek raiser intensity |
| 2 | AU12 | Lip corner puller intensity |
| 3 | AU14 | Dimpler intensity |
| 4 | AU25 | Lips part intensity |
| 5 | AU26 | Jaw drop intensity |
| 6 | AU45 | Blink intensity |
| 7 | eye-open-right | Landmark distance 159–145 |
| 8 | eye-open-left | Landmark distance 386–374 |
| 9 | eye-raise-right | Landmark distance 105–159 |
| 10 | eye-raise-left | Landmark distance 334–386 |
| 11 | mouth-open | Landmark distance 13–14 |
| 12 | mouth-width | Landmark distance 61–291 |
| 13 | jaw-open | Landmark distance 13–152; geometric proxy |

The versioned measurement convention uses OpenFace AU intensities in `[0,5]`
and MediaPipe's 478-point landmarks. Geometry is the 3D Euclidean distance
between the listed points divided by inter-iris distance (468–473). Convert
coordinates to `(x*width, y*height, z*width)` before measuring. Left/right are
anatomical on an unmirrored image. MediaPipe blendshape coefficients are not
interchangeable with OpenFace AU intensities. Head pose is not part of these
14 inputs or the final 42 features.

**Seven binary AU-presence flags are also required**, in the same AU order.
They are numerical metadata, not seven additional measured feature channels.
Only frames where that AU's presence is 1 contribute to its summary. AU
intensity alone cannot recover this rule reliably; presence must not be guessed
from whether intensity exceeds zero. Geometry uses all usable frames.

For each measurement the output contains **mean, population variance, entropy**,
in that order, yielding 14 × 3 = 42 values. `FEATURES` exposes the exact column
names. Entropy is `-sum(p*ln(p))`, where `p=x/sum(x)` over positive samples;
it depends on frame/sample count and is not histogram entropy or entropy rate.
An AU with no active frames produces three zeros. Entirely failed observations
must be withheld, not replaced with a zero facial vector.

`summarize` requires at least 30 usable frames. The frontend must also preserve
the extraction quality policy: one face; OpenFace success/confidence >=0.8;
matching OpenFace/MediaPipe face boxes (IoU >=0.3); inter-iris distance >=10 pixels;
and >=80% usable coverage of the selected interval. The existing offline
extractor rejects clips with multiple faces. Avoid mixing different people.

```python
from exports.facial import summarize

face = summarize(usable_measurements, usable_au_presence)  # [42]
```

This module reproduces our numerical aggregation, but **the fusion weights were
trained on supplied UFNet prompted-smiling summaries**. Exact parity between the
frontend extractor and UFNet is unverified. A five-second capture interval is
an engineering setting, not a duration learned by this model. Keep extractor
versions, sampling rate, window duration, AU presence policy, and temporal
smoothing consistent. The existing OpenFace implementation smooths using
neighboring frames; a streaming frontend needs a defined buffering policy.

## 3. Eleven-token attention fusion

`weights/fusion_gallery.pt` contains three frozen networks, each with **4,481
parameters**, plus the training scalers and feature schema. File size: 76,225
bytes. It expects exactly ten embeddings per observation, without padding the
gallery itself or repeating vectors to fill it. Individual embeddings may come
from partial keystroke sequences because those were included in training.

One 42-value face vector and ten 128-value typing vectors are projected to 16
dimensions each. Two-head self-attention processes **all 11 tokens jointly**.
After attention, the updated face token is concatenated with the mean of the
ten updated typing tokens. The MLP is `32 -> 16 -> 1`, with GELU. No gallery
positions or cross-session chronology are encoded; typing-token order does not
change the output except for floating-point rounding.

Each network applies sigmoid to its logit; the three probabilities are averaged.
Output columns are `[p_control, p_impaired]`. Classification uses `p_impaired >=
0.5`. Probabilities are uncalibrated and the positive training labels were PD.
All dropout is off during inference. **Pass unstandardized inputs**: this module
applies the saved training scalers exactly once.

```python
from exports.fusion import FrozenFusion

fusion = FrozenFusion()
# embeddings[:10] must be ten distinct sequences from this user.
probabilities = fusion.predict_proba(face[None, :], embeddings[:10][None, :, :])
p_impaired = float(probabilities[0, 1])
```

For more than ten sequences, create disjoint complete galleries and classify
each. A later batch-level summary can average their probabilities. Keep
timestamps and subject linkage outside the neural model. The export neither
maintains a personal baseline nor interprets longitudinal change.

## Experiment result and deployment recommendation

This is an **experimental export**, not a replacement selected over the existing
two-token model. Participant-level metrics average gallery probabilities within
each fixed pair:

| Split | Participants | Accuracy | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Training | 218 | 93.1% | 97.0% | 89.0% | 92.8% |
| Validation | 31 | 74.2% | 71.4% | 71.4% | 71.4% |
| Test | 64 | 84.4% | 86.2% | 80.6% | 83.3% |

There are 829/128/236 train/validation/test galleries. The existing two-token
reference, rescored on identical retained inputs, reaches 93.1%/80.6%/82.8%
accuracy. Gallery attention improves one test prediction but worsens validation.
The reference was trained on more windows, so this is not a controlled
single-factor architecture comparison. Test participants were already inspected
in earlier experiments; synthetic pairs do not establish real paired performance.

Keep these modules separately versioned and deploy them as **one pinned pipeline
release** using `manifest.json`. The manifest records checkpoint/source hashes,
feature order, input dimensions, normalization ownership, and evaluation scope.
Do not mix an arbitrary TypeNet checkpoint or facial measurement convention with
these fusion weights. The facial component includes code and measurement rules,
not just model weights; changing those rules changes the input representation.
