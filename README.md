# mciFaceNet

CURRENT STATUS: FINALIZED PIPELINE UNTIL MORE DATA IS AQUIRED THROUGH INSTITUTIONAL CONNECTIONS. EXPORT TBC

Classify recordings from UFNet's 42 extracted facial features and export a
probability for each observation. Inference needs no personal baseline, recording
dates, or previous predictions.

The output classes are `control` and `impaired`, with one combined positive
target. The current experiment maps explicit PD-positive labels to `impaired`;
its reported results measure discrimination of those source labels. They do not
establish performance for MCI or cognitive decline.

## Run

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --extra train
uv run mcifacenet fetch
uv run --extra train mcifacenet train --output runs/experiment-1
uv run mcifacenet predict \
  --model runs/experiment-1/classifier.json \
  --input features.csv \
  --output predictions.json
```

`fetch` downloads the pinned public features, participant splits, configuration,
reference script, and license, verifying SHA-256 checksums. Training requires a
new output directory. Prediction also refuses to overwrite an existing output.
Downloaded data and experiment outputs are gitignored.

The input CSV needs all 42 named features listed in
[`model.py`](src/mcifacenet/model.py): `mean`, `var`, and `entropy` for seven action
units and seven facial geometry measures. Column order does not matter. Metadata
such as IDs and labels is ignored by the classifier. Missing, nonfinite, or
unexpected `smile_` features are rejected. Each row represents one recording,
not one frame. A separate [video extraction pipeline](docs/facial_extraction.md)
now exports frame-level signals, clip summaries, and configurable window summaries.
Its numerical compatibility with UFNet's unreleased extractor is unverified;
matching column names alone does not establish compatibility with saved weights.
The [YouTubePD-only experiment](docs/youtubepd_classification.md) retrains both
original classifier architectures on quality-passing five-second video windows.

Predictions contain `p_control`, `p_impaired`, a class decision at the fixed 0.5
threshold, the source row/ID, and a model fingerprint. The Python API returns
probabilities in `[control, impaired]` order:

```python
import csv
from mcifacenet import FacialClassifier

model = FacialClassifier.load("runs/experiment-1/classifier.json")
with open("features.csv", newline="") as stream:
    probabilities = model.predict_proba(csv.DictReader(stream))
```

The JSON model includes feature order, scaling, coefficients, and calibration.
Inference uses only the Python standard library; training libraries are optional.

## Experiment protocol

The facial-only experiment adapts
[UFNet](https://github.com/ROC-HCI/UFNet/tree/5ece2c65ba184faccf6c8cdccdc03132427c464b),
including its extracted [YouTubePD](https://github.com/samwli/YouTubePD-data)
features. It is not an exact reproduction of the multimodal paper's results.

- Use only explicit `yes`/`no` UFNet labels, excluding 39 ambiguous or missing
  labels. The derived `Diagnosis` column is not used.
- Keep the author's participant IDs disjoint across training (863 recordings),
  validation (340), calibration (129), and test (313). Reserve the published
  calibration participants from training.
- Fit standardization on training data only. Train UFNet's published shallow
  scalar-logit dropout architecture and settings, with training-only SMOTE, plus
  an unaugmented logistic-regression reference.
- Select the neural epoch and final candidate using raw validation log loss.
  Fit Platt calibration on the separate calibration participants. Neither test
  set is used for model selection or threshold tuning.
- Evaluate all 251 rows of `youtube_PD_features_updated.csv` separately; no
  YouTubePD observations are used for fitting. The two released YouTube feature
  CSVs describe the same clips and are not combined.

For UFNet's single-logit dropout layer, inference uses the exact expected
probability, `d / 2 + (1 - d) * sigmoid(logit / (1 - d))`, where `d` is the
dropout probability. This replaces sampling 1,000 dropout trials with a
deterministic result. It does not reproduce uncertainty estimates from those
trials.

Each run writes the selected `classifier.json`, both calibrated candidates in
`models/`, per-observation `predictions.csv`, and `metrics.json`. The accompanying
`protocol.json`, `selection.json`, and `training_history.json` record source and
code hashes, dependency versions, exclusions, splits, and model selection.
Metrics include discrimination, calibration, confusion matrices, and a constant
training-prevalence reference. Training and calibration-set metrics describe
fitting performance, not held-out evaluation.

## First experiment

The fixed protocol selected logistic regression by validation log loss (0.419,
versus 0.439 for the UFNet candidate). Results below use its calibrated output
and the predefined 0.5 threshold:

| Evaluation data | Recordings | Accuracy | Recall | Precision | F1-score |
| --- | ---: | ---: | ---: | ---: | ---: |
| UFNet validation | 340 | 82.1% | 65.8% | 77.3% | 71.1% |
| Held-out UFNet test | 313 | 76.4% | 53.3% | 60.0% | 56.5% |
| YouTubePD clips | 251 | 56.2% | 82.8% | 32.4% | 46.6% |

The held-out UFNet confusion matrix has 191 true negatives, 32 false positives,
42 false negatives, and 48 true positives. YouTubePD has 100 false positives
among 193 negative clips. Its probability calibration transfers poorly, so these
scores should not be treated as validated clinical probabilities.

UFNet participant-level evaluation averages each person's recording probabilities
before scoring: accuracy 79.9%, recall 56.5%, precision 63.9%, and F1-score 60.0%
across 259 participants. This aggregation is an evaluation summary, not a
longitudinal baseline. YouTubePD IDs identify clips; person identities and identity
overlap with UFNet are unverified. Its metrics therefore do not demonstrate
independent-subject generalization.

Reloading the exported classifier reproduced predictions for all 1,896 eligible
recordings exactly. Local artifacts from this run are in
`runs/facial_classifier_v1/`; they are not bundled in the repository.

## Facial and typing fusion experiment

The optional fusion experiment combines the 42 facial features with the frozen
128-dimensional TypeNet embedding of one typing window of up to 50 keys. Faces and typing
windows come from different people and are paired using their shared binary
label. These are **artificial multimodal cases**, not recordings of the same
person. No labels are required by the exported inference model.

```sh
uv run --extra train mcifacenet fusion-train \
  --typing-run /path/to/registered/typing/runs/embeddings \
  --face-model runs/facial_classifier_v1/classifier.json \
  --output runs/facial_typing_fusion_v1

uv run --extra train mcifacenet fusion-predict \
  --model runs/facial_typing_fusion_v1/models/attention.pt \
  --input runs/facial_typing_fusion_v1/example_input.npz \
  --output runs/facial_typing_fusion_v1/example_predictions.json
```

The typing adapter reads the registered real-only embedding cache, checks hashes
against its parent cache and encoder, and retains its original participant
assignments. The existing typing classifier export must be in the sibling
`exports/` directory with its verification file. The source run supplies the
registered relative paths for its encoder and parent data. It is read only.

This run uses 221/32/64 typing participants in train/validation/test, alongside
699/264/259 facial participants. Typing sources are neuroQWERTY and the CoNLL
online typing cohort; online PD labels are self-reported. Facial calibration
participants remain outside fusion training. Anonymous identity overlap across
source studies cannot be verified.

Both modality projections have width 32. The attention network applies four-head
self-attention to the two modality tokens, followed by a small classification
head. The concatenation control uses the same projections and classification
head without the attention block. Both train for 64 epochs with AdamW, with
4,096 newly sampled same-label pairs per epoch and three fixed seeds. Sampling
and standardization balance participants so prolific typists do not dominate.
No TypeNet encoder weights are updated and no synthetic keystroke windows are
generated.

Each seed's checkpoint is selected by mean validation F1, breaking ties with
log loss. All three selected checkpoints contribute equally to the exported
probability ensemble. Evaluation uses 20 fixed pairing draws per split, each
containing every facial recording once. The decision threshold remains 0.5.

| Model | Validation accuracy | Validation F1 | Test accuracy | Test F1 |
| --- | ---: | ---: | ---: | ---: |
| Existing facial classifier | 82.1% | 71.1% | 76.4% | 56.5% |
| Existing typing classifier | 61.8% | 41.0% | 70.6% | 51.4% |
| Mean of existing probabilities | 80.8% | 67.4% | 81.6% | 64.6% |
| Concatenation network | 79.4% | 68.0% | 81.5% | 67.9% |
| Attention network | 79.7% | 68.3% | 81.5% | 67.3% |

These are means over the artificial pairing draws. Positive-class metrics for
the attention ensemble are:

| Pairing evaluation | Accuracy | Recall | Precision | F1-score |
| --- | ---: | ---: | ---: | ---: |
| UFNet validation + typing validation | 79.7% | 65.3% | 71.7% | 68.3% |
| UFNet test + typing test | 81.5% | 65.9% | 68.7% | 67.3% |
| YouTubePD + typing test | 66.4% | 84.3% | 39.4% | 53.7% |

Attention improves on the existing facial classifier's synthetic test F1 but
does not beat concatenation there, and facial-only remains strongest on
validation. The experiment does not establish an advantage for attention.
The individual-modality comparisons above use the same artificial cases, so
their class proportions follow the facial dataset. Native typing participant
metrics are stored separately in the report.

The run saves both ensemble exports, all pair identities and predictions,
comparison metrics including precision/recall/F1, training histories, selected
epochs, source snapshots, and hashes in `runs/facial_typing_fusion_v1/`.
`comparison.csv` provides a compact table; `metrics.json` includes variation
across pairing draws. That variation is not a participant confidence interval.
Repeated pairings do not increase the number of independent people.

For custom inference, supply an NPZ containing `face` with shape `[N,42]` in the
export's named feature order, and `typing` with shape `[N,128]` from the exact
frozen encoder identified in the artifact. Both arrays must have aligned rows.
The exported fusion model includes preprocessing and all three neural heads;
it expects extracted features and does not bundle the raw typing encoder or
video extractor into the inference artifact. The separate video extraction
pipeline is documented above. Predictions are uncalibrated model probabilities.

The underlying test data had already been inspected in earlier experiments.
This is an exploratory follow-up, and all pairing evaluations depend on known
labels when constructing artificial cases. Validating fusion for a real user
requires genuinely paired facial and typing test observations.

## Fixed participant fusion experiment

The second experiment assigns each typist to exactly one same-label facial
participant, without sharing either person across synthetic participant pairs.
Every cached typing window appears once in the fixed dataset and once per
training epoch. Only row order changes between epochs; a window is never moved
to another face. When a matched facial person has several recordings, their
shuffled recordings are cycled across that typist's windows and then fixed.
Facial features can therefore repeat within the same participant pair.

```sh
uv run --extra train mcifacenet fusion-train-fixed \
  --typing-run /path/to/registered/typing/runs/embeddings \
  --previous-run runs/facial_typing_fusion_v1 \
  --output runs/facial_typing_fusion_fixed_v1
```

| Split | Participant pairs | Facial recordings used | Fixed window pairs |
| --- | ---: | ---: | ---: |
| Training | 221 | 290 | 8,979 |
| Validation | 32 | 42 | 1,366 |
| Test | 64 | 74 | 2,571 |

All eligible typing windows are retained. The source cache includes 9,459 full
50-key windows and 3,457 padded partial windows. Window counts are not counts
of independent people. One facial training participant with conflicting labels
is ineligible for matching; unused facial participants remain outside this run.
The original participant splits, calibration exclusion, frozen encoder, model
architectures, three seeds, optimizer settings, and 64-epoch budget are retained.

Training weights each window's loss inversely by its typist's window count, so
people contribute equally without oversampling. Facial scaling uses only the
selected training recordings. Checkpoint selection uses validation F1 after
averaging probabilities within each participant pair, with log loss breaking
ties. Selection, loss weighting, cohort composition, and the number of updates
per epoch differ from the earlier experiment; this is not a single-factor
ablation. The attention checkpoints were selected at epochs 2, 6, and 15.

Attention ensemble results at threshold 0.5, **after averaging probabilities
within each participant pair**:

| Split | Accuracy | Recall | Precision | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Training | 100.0% | 100.0% | 100.0% | 100.0% |
| Validation | 84.4% | 78.6% | 84.6% | 81.5% |
| Test | 79.7% | 86.2% | 73.5% | 79.4% |

The same model's individual-window results are:

| Split | Accuracy | Recall | Precision | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Training | 99.9% | 100.0% | 99.8% | 99.9% |
| Validation | 84.4% | 78.2% | 84.6% | 81.3% |
| Test | 75.2% | 75.9% | 73.0% | 74.4% |

On these same fixed participant pairs, the previous attention model scores
78.1% test accuracy and 72.0% F1; the newly trained concatenation control scores
79.7% and 77.2%. At window level, concatenation scores 80.0% accuracy and 77.9%
F1, exceeding attention's window scores. Prior models retain their original,
larger facial training pools. These comparisons do not establish a consistent
advantage for attention across evaluation units.

Perfect training classification and lower held-out scores indicate overfitting.
There are only 32 validation and 64 test participant pairs, using one fixed
assignment. Seed ensembling does not measure uncertainty over alternative
partner assignments. These previously inspected test participants and artificial
label-matched pairs do not establish real matched-patient performance.

Artifacts in `runs/facial_typing_fusion_fixed_v1/` include both exports,
train/validation/test pair identities, all model predictions, participant and
window metrics, source snapshots, full training histories, selection hashes,
and `verification.json`. Use the existing `fusion-predict` command with the new
`models/attention.pt` for feature-input inference. Aggregation is an evaluation
step; the export still returns one probability per input window pair.

## Regularization and sliding-window attention

```sh
uv run --extra train mcifacenet fusion-regularize \
  --typing-run /path/to/registered/typing/runs/embeddings \
  --fixed-run runs/facial_typing_fusion_fixed_v1 \
  --output runs/facial_typing_fusion_regularized_v1
```

This experiment retains the exact saved participant and window assignments.
Four prespecified variants compare regularized two-token fusion, full attention
over each ordered typing sequence, and local attention with spans of 9 or 17
keystrokes. Local attention uses a symmetric band mask, following the local
attention idea described in [Longformer](https://arxiv.org/abs/2004.05150).
It uses a dense masked implementation for sequences of at most 50 keys, with
no claim of Longformer's sparse computational efficiency.

Regularization uses width 16/two heads, 35% attention-weight and residual/head
dropout, 20% input dropout, a 15% chance of dropping one modality during
training, AdamW weight decay 0.1, and label smoothing 0.05. The learning rate
is 0.0005, with gradient clipping at 1.0. Checkpoints and the final candidate
are selected by participant validation log loss, which penalizes confident
mistakes; early stopping uses patience eight and minimum improvement 0.0001,
with a maximum of 48 epochs. Each candidate averages three fixed seed models.
The prediction threshold remains 0.5; validation/test labels are never smoothed.

Sequence variants retain the frozen 128-coordinate typing embedding and add
an attention-pooled representation of the five measured per-key features:
hold, inter-key, press and release latencies, and normalized key code. Original
source caches are hash-checked and aligned exactly with the registered embedding
rows. Attention operates within each original window; it never treats embedding
coordinates or arbitrary participant ordering as a temporal sequence. Facial
inputs remain recording summaries, without fabricated frame-level dynamics.

All sequence variants share the same inputs and parameter count. Fixed sinusoidal
positions preserve key order. A training-only scaler balances participants and
ignores padding; standardized key features are clipped to ±5. Padded keys are
excluded from attention and pooling. The model's existing face/embedding scalers
are retained from the fixed-pair run. Full and local masks use PyTorch's
[MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html).

The regularized exports use format version 2 and the same `fusion-predict`
command. Sequence models additionally require `sequence` with shape `[N,50,5]`
and integer `lengths` with shape `[N]`, ranging from 1 to 50, in the input NPZ.
Features must follow the export's `sequence_features` order and the source
TypeNet feature definitions; shorter windows are padded. There is one prediction
per original window pair, followed by separate participant probability averaging
for evaluation. No new cross-person pairings or overlapping training examples
are generated by the sliding attention mask.

All results below average window probabilities within each participant pair
(221 training, 32 validation, 64 test). The previous fixed-pair models are
rescored on the exact same inputs.

| Attention variant | Training accuracy | Validation accuracy | Test accuracy | Test F1 |
| --- | ---: | ---: | ---: | ---: |
| Previous fixed-pair attention | 100.0% | 84.4% | 79.7% | 79.4% |
| Regularized two-token fusion | 93.2% | 81.3% | 82.8% | 81.4% |
| Full keystroke-sequence attention | 92.3% | 81.3% | 84.4% | 82.8% |
| Sliding attention, span 9 | 92.3% | 81.3% | 84.4% | 82.8% |
| Sliding attention, span 17 | 92.3% | 81.3% | 84.4% | 82.8% |

Regularized two-token fusion was selected **among the four new candidates** by
validation log loss before test scoring. Its complete participant metrics are:

| Split | Accuracy | Recall | Precision | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Training | 93.2% | 96.0% | 89.7% | 92.8% |
| Validation | 81.3% | 78.6% | 78.6% | 78.6% |
| Test | 82.8% | 82.8% | 80.0% | 81.4% |

Its individual-window accuracy/recall/precision/F1 are
92.6/94.7/89.3/92.0% for training, 81.3/77.8/78.6/78.2% for validation,
and 80.3/76.3/81.1/78.6% for test. The earlier model still has stronger validation
accuracy and slightly better validation log loss. The new regularized model's
train–test accuracy gap falls from 20.3 to 10.4 percentage points, with two more
test participant pairs and one fewer validation pair correctly classified.
This does not establish an overall winner or justify test-based promotion.

Full and sliding sequence attention make identical participant classifications,
with only small differences between individual-window probabilities. There is
no observed advantage for restricting the attention span. The source sequence
variants add measured keystroke features as well as changing attention, so their
comparison with the two-token model is not solely a masking ablation.

The new checkpoints stopped after 15–21 epochs. All four exports and detailed
window/participant metrics are stored in `runs/facial_typing_fusion_regularized_v1/`.
`selection.json` identifies the candidate selected by validation, and
`verification.json` records independent metrics, checkpoint reconstruction,
pairing preservation, source checks, and CLI/export parity.

## Checks and attribution

```sh
uv run --extra train pytest -q
```

Tests cover feature validation, metadata exclusion, deterministic probability
inference, the analytical dropout expectation, portable exports, source integrity,
participant separation, and training-only scaling. Source-data checks need
`fetch`; the completed-run scaling check needs the first experiment artifact.

Source URLs and checksums are in [`sources.json`](src/mcifacenet/sources.json).
UFNet attribution and its license are retained in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
