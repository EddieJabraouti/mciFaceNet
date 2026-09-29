# mciFaceNet

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
not one frame; raw-video feature extraction is not implemented.

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
128-dimensional TypeNet embedding of one 50-key typing window. Faces and typing
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
it expects extracted features and does not contain the raw typing encoder or
video feature extractor. Predictions are uncalibrated model probabilities.

The underlying test data had already been inspected in earlier experiments.
This is an exploratory follow-up, and all pairing evaluations depend on known
labels when constructing artificial cases. Validating fusion for a real user
requires genuinely paired facial and typing test observations.

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
