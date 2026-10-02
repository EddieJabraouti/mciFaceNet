# Facial-only classification trained on YouTubePD

Run both original facial-classifier architectures from scratch on the extracted
five-second YouTubePD windows:

```sh
uv run --extra train mcifacenet youtube-train \
  --output runs/youtubepd_facial_only_v1
```

The command uses the extraction under `data/processed/youtubepd_facial_v1`, its
download manifest, and the hyperparameter recipe saved in
`runs/facial_classifier_v1/protocol.json`. Those paths can be overridden with
`--features-dir`, `--manifest`, and `--recipe`. The output directory must be new.
No UFNet feature rows or previously trained weights enter model fitting.

## Data and partitions

Use only windows that pass the extractor's frozen automatic quality rules:
416 windows from 230 clips and 217 distinct YouTube video IDs. Exclude 62 flagged
windows. A video ID is **not a participant ID**. The release does not supply a
reliable identity mapping across different uploads, so participant independence
cannot be established.

Preserve the main spreadsheet's published train/validation/test assignments.
Assign supplementary negative video groups approximately 60/20/20, using seeds
462 and 463. Reserve 20% of training video groups for calibration, stratified by
label, using seed 464. All segments/windows from a source video stay together,
including overlapping clips. Assignments are saved before fitting. Exact feature
duplicates and repeated source-file hashes across partitions are rejected.

| Partition | Windows | PD-positive | Negative | Clips | Source videos |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 181 | 45 | 136 | 103 | 101 |
| Validation | 93 | 27 | 66 | 49 | 45 |
| Calibration | 54 | 11 | 43 | 29 | 26 |
| Test | 88 | 11 | 77 | 49 | 45 |

Training-only standardization is shared by both candidates. Logistic regression
uses C=1, L-BFGS, and no oversampling. The shallow network retains the original
42-to-1 linear layer, scalar-logit dropout (0.1066175644), sigmoid, SGD recipe,
and 64 epochs. SMOTE applies only to its training partition: 181 real windows
become 272 training samples. Reported training metrics use the 181 real windows.

Raw validation log loss chooses the neural epoch and final candidate. The
shallow model was selected at epoch 45. Separate Platt calibration is fitted on
the calibration partition. The decision threshold remains 0.5; test outcomes
do not select settings. Source `y`/`n` labels map to the existing binary target.

## First fixed run

Results below are percentages, computed per window **before calibration**.
Precision, recall, and F1 refer to the PD-positive class.

| Model | Partition | Accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Logistic regression | Training | 88.40 | 83.33 | 66.67 | 74.07 |
| Logistic regression | Validation | 75.27 | 61.11 | 40.74 | 48.89 |
| Logistic regression | Test | 77.27 | 23.53 | 36.36 | 28.57 |
| Shallow network | Training | 75.69 | 50.67 | 84.44 | 63.33 |
| Shallow network | Validation | 70.97 | 50.00 | 62.96 | 55.74 |
| Shallow network | Test | 62.50 | 21.05 | 72.73 | 32.65 |

**After calibration, both candidates predict every window as control at 0.5.**
Their training/validation/test accuracies become 75.14% / 70.97% / 87.50%, with
positive precision, recall, and F1 all 0%. The 87.50% test accuracy equals always
predicting control (77 of 88 test windows). Undefined precision when no positives
are predicted is reported as zero. The calibrated exports therefore do not
demonstrate useful positive-class detection at the fixed threshold.

The raw logistic test confusion matrix is `[[64,13],[7,4]]`; the raw shallow
matrix is `[[47,30],[3,8]]`. Calibrated matrices are `[[77,0],[11,0]]`.
Rows are true negative/positive classes; columns are predicted negative/positive.

The run also reports clip- and source-video-level metrics after averaging window
probabilities. These are not participant-level results. The test has only eight
positive source videos, and correlated windows do not increase that independent
sample count. Different frame rates, interview conditions, possible repeated
people across uploads, and the extractor's unverified UFNet numerical parity
limit interpretation. These results assess source PD labels, not MCI detection.

## Artifacts and verification

The run writes `protocol.json`, `splits.csv`, `exclusions.json`,
`selection.json`, `training_history.json`, `metrics.json`, `comparison.csv`, and
`predictions.csv`. Four candidate exports in `models/` preserve raw and calibrated
variants. `classifier.json` is the selected calibrated shallow model, retained
for reproducibility rather than evidence of a useful deployment threshold.

The 70-test suite passed. Independent verification reproduced all 1,664 saved
probabilities across the four exports, recomputed accuracy/precision/recall/F1,
checked source-video/clip/file-hash separation, and confirmed scaling used only
training windows. No existing model artifacts were overwritten.
