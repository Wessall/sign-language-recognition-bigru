# BiGRU — Sign Language Gesture Recognition

A Bidirectional Gated Recurrent Unit (BiGRU) model for skeletal sequence-based sign language recognition, trained on 94,477 sequences spanning 250 gesture classes. This repository contains the full training pipeline, evaluation artifacts, saved model weights, TFLite export, and analysis notebooks.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Per-Class Analysis](#per-class-analysis)
- [Error Analysis](#error-analysis)
- [Project Structure](#project-structure)
- [Saved Artifacts](#saved-artifacts)
- [Notebooks](#notebooks)
- [Reproducing the Experiment](#reproducing-the-experiment)
- [Known Limitations](#known-limitations)

---

## Overview

This model classifies sign language gestures from skeletal landmark sequences. Each input is a variable-length sequence of body and hand pose landmarks, fixed to a maximum of 384 frames, with 118 selected landmark nodes producing 708 feature channels per frame. The model processes each sequence in both forward and backward directions simultaneously, capturing contextual information across the full temporal span of each gesture.

---

## Dataset

| Property              | Value                                               |
|-----------------------|-----------------------------------------------------|
| Total sequences       | 94,477                                              |
| Unique classes        | 250 sign vocabulary                                 |
| Landmark nodes        | 118 selected                                        |
| Feature channels      | 708 per frame                                       |
| Max sequence length   | 384 frames                                          |
| Input tensor shape    | (Batch, 384, 708)                                   |
| Training split        | 75,581 samples (80%) — avg 302.3 per class          |
| Validation split      | 9,448 samples (10%) — avg 37.8 per class            |
| Test split            | 9,344 samples — avg 37.8 per class                  |
| Class balance         | Very balanced — min 30, max 41 per class (test set) |

---

## Model Architecture

| Property               | Value                              |
|------------------------|------------------------------------|
| Core unit              | Bidirectional GRU                  |
| Gating mechanism       | 3 gates: reset, update, new        |
| Positional encoding    | None                               |
| Trainable parameters   | 2.44M                              |
| Saved formats          | SavedModel (.keras) + TFLite       |
| Validation-to-test gap | 0.92%                              |

The GRU cell uses three gating operations — reset, update, and new — compared to four in an LSTM cell. This reduces parameter count while maintaining comparable representational capacity for sequence modeling at this scale. The bidirectional wrapper processes each input sequence in both directions, allowing the model to use future frame context when classifying any given frame.

---

## Performance

Evaluated on the 9,344-sample held-out test set.

| Metric          | Value  |
|-----------------|--------|
| Accuracy        | 81.74% |
| Top-5 Accuracy  | 93.58% |
| Macro F1        | 0.8151 |
| Weighted F1     | 0.8173 |
| Macro ROC-AUC   | 0.9880 |
| Macro PR-AUC    | 0.8423 |
| Misclassified   | 1,706  |

The top-5 accuracy of 93.58% confirms that the correct label is within the model's top five predictions for the vast majority of samples, which is favorable for candidate-list assistive applications. The validation-to-test gap of 0.92% indicates stable generalization with no significant overfitting.

---

## Per-Class Analysis

**Best performing classes:**

| Class       | Precision | Recall | F1   |
|-------------|-----------|--------|------|
| horse       | 1.00      | 0.95   | 0.97 |
| callonphone | 0.95      | 1.00   | 0.97 |
| cowboy      | 0.97      | 0.95   | 0.96 |
| brown       | 0.97      | 0.95   | 0.96 |
| frog        | 0.95      | 0.97   | 0.96 |
| uncle       | 0.97      | 0.95   | 0.96 |
| lion        | 0.97      | 0.95   | 0.96 |

**Most difficult classes:**

| Class    | Precision | Recall | F1   | Primary Issue                        |
|----------|-----------|--------|------|--------------------------------------|
| beside   | 0.33      | 0.42   | 0.37 | Weak spatial trajectory cue          |
| awake    | 0.47      | 0.45   | 0.46 | Near-identical motion to "wake"      |
| wake     | 0.49      | 0.47   | 0.48 | Near-identical motion to "awake"     |
| pen      | 0.61      | 0.57   | 0.59 | Handshape overlap with "pencil"      |
| bedroom  | 0.70      | 0.50   | 0.58 | Compound sign confusion with "bed"   |

Mean per-class accuracy across all 250 classes: **81.59%**. Approximately 12 classes fall below an F1 of 0.70.

---

## Error Analysis

**Top confused pairs on the test set:**

| True Class | Predicted | Count | Root Cause                              |
|------------|-----------|-------|-----------------------------------------|
| awake      | wake      | 18    | Identical handshape, subtle timing diff |
| wake       | awake     | 18    | Symmetric — same underlying issue       |
| bedroom    | bed       | 11    | Compound sign subset overlap            |
| sleepy     | sleep     | 10    | Timing-based disambiguation failure     |
| give       | gift      | 10    | Motion trajectory similarity            |
| lips       | mouth     | 8     | Facial location overlap                 |
| pencil     | pen       | 7     | Fine-grained handshape similarity       |
| pen        | pencil    | 7     | Fine-grained handshape similarity       |

The awake/wake confusion (18 errors in each direction) is a structural vocabulary problem. These two signs share the same handshape and facial location, differing only in a subtle temporal extension. This is not resolvable through architecture changes alone and requires data-level intervention — specifically, augmenting these two classes with speed-varied examples.

The pencil/pen confusion is a handshape disambiguation problem. The distinguishing feature between these two signs is finger configuration rather than motion trajectory, which falls outside what the BiGRU's temporal modeling captures.

---

## Project Structure

```
.
|-- organize_project.py
|-- structure.txt
|
+-- archives/
|   +-- final_model_BiGRU.zip              # Complete packaged model archive
|
+-- comparisons/
|   +-- model_comparison_summary.csv
|
+-- experiments/
|   +-- exp_001/
|       |-- metadata.json                  # Experiment config and hyperparameters
|       |
|       +-- checkpoints/
|       |   +-- best/                      # Best checkpoint (ckpt-159)
|       |   +-- last/                      # Final checkpoints (ckpt-164, 165)
|       |
|       +-- data/
|       |   +-- train_split.csv
|       |   +-- val_split.csv
|       |   +-- test_split.csv
|       |
|       +-- outputs/
|           +-- logs/                      # Training logs (7 sessions, April 2026)
|           +-- metrics/                   # Full evaluation metrics
|           +-- plots/                     # All visualization outputs
|           +-- predictions/
|           |   +-- test_predictions.csv
|           +-- reports/
|               +-- classification_report.txt
|               +-- evaluation_summary.txt
|
+-- models/
|   +-- bigru/
|       |-- label_map.json                 # Class index to label mapping (250 classes)
|       |-- model.tflite                   # TFLite export for edge/mobile deployment
|       |-- model_info.json                # Architecture metadata
|       |
|       +-- saved_model/
|           |-- fingerprint.pb
|           |-- saved_model.pb
|           +-- variables/
|               |-- variables.data-00000-of-00001
|               +-- variables.index
|
+-- notebooks/
    |-- 02-bigru-train.ipynb               # Training pipeline
    +-- bigru-evaluation.ipynb             # Evaluation and visualization
```

---

## Saved Artifacts

### Metrics — `experiments/exp_001/outputs/metrics/`

| File                        | Description                                              |
|-----------------------------|----------------------------------------------------------|
| `evaluation_summary.csv`    | Top-level metrics: accuracy, F1, AUC, sample counts      |
| `classification_report.csv` | Per-class precision, recall, F1, and support             |
| `per_class_accuracy.csv`    | Per-class accuracy for all 250 classes                   |
| `pr_auc_per_class.csv`      | Per-class PR-AUC values                                  |
| `roc_auc_per_class.csv`     | Per-class ROC-AUC values                                 |
| `confused_pairs.csv`        | Top confusion pairs with error counts                    |
| `misclassified_samples.csv` | All 1,706 misclassified samples with confidence scores   |
| `hard_examples.csv`         | 50 lowest-confidence correct predictions                 |
| `training_history.csv`      | Per-epoch loss and accuracy across all training sessions |
| `inference_performance.csv` | Latency and throughput benchmarks                        |
| `model_params.txt`          | Parameter count breakdown                                |
| `model_size.csv`            | Model file size statistics                               |
| `test_results.csv`          | Full test set predictions with confidence scores         |
| `dataset_report.txt`        | Dataset statistics for this experiment                   |

### Plots — `experiments/exp_001/outputs/plots/`

| File                              | Description                                      |
|-----------------------------------|--------------------------------------------------|
| `BiGRU_training_history.png`      | Loss and accuracy curves across training epochs  |
| `BiGRU_architecture.png`          | Visual diagram of the model architecture         |
| `confusion_matrix_normalized.png` | Row-normalized confusion matrix (250 x 250)      |
| `confusion_matrix_raw.png`        | Raw count confusion matrix                       |
| `confused_pairs.png`              | Bar chart of top confusion pairs                 |
| `per_class_accuracy.png`          | Per-class accuracy distribution                  |
| `metrics_distribution.png`        | Precision, recall, and F1 distributions          |
| `roc_curves.png`                  | Macro and per-class ROC curves                   |
| `precision_recall_curves.png`     | Macro and per-class PR curves                    |
| `data_split_distribution.png`     | Train/val/test class distribution                |

---

## Notebooks

**`02-bigru-train.ipynb`** covers data loading from landmark CSV splits, sequence padding and masking, BiGRU model definition, training with callbacks (early stopping, checkpoint saving), and export to SavedModel and TFLite formats.

**`bigru-evaluation.ipynb`** covers loading the saved model, running inference on the test split, computing all evaluation metrics, generating confusion matrices, extracting hard examples and misclassified samples, and producing all plots saved under `outputs/plots/`.

---

## Reproducing the Experiment

The experiment is recorded under `experiments/exp_001`. Seven training sessions were logged between April 6 and April 13, 2026.

**To run evaluation from saved weights:**

1. Load the SavedModel from `models/bigru/saved_model/` or the TFLite model from `models/bigru/model.tflite`.
2. Use `models/bigru/label_map.json` to map predicted class indices back to sign names.
3. Run `bigru-evaluation.ipynb` using the test split at `experiments/exp_001/data/test_split.csv`.

**To resume or retrain from the best checkpoint:**

1. Restore from `experiments/exp_001/checkpoints/best/ckpt-159`.
2. The training data index is at `experiments/exp_001/data/train_split.csv`.
3. Experiment configuration and hyperparameters are in `experiments/exp_001/metadata.json`.

---

## Known Limitations

**awake / wake confusion.** These two signs share handshape, facial location, and movement onset, differing only in a temporal extension that is insufficiently represented in the current training distribution. Targeted data augmentation with speed-varied examples of both classes is required to address this.

**pencil / pen confusion.** The distinction between these signs relies on finger configuration rather than motion trajectory. A handshape-specific auxiliary loss or dedicated feature injection would be needed to improve accuracy on this pair.

**No temporal position encoding.** The BiGRU has no mechanism to explicitly encode the relative position of frames within the sequence, which limits disambiguation of signs whose primary difference is timing rather than motion type.

**Single-dataset training.** This model was trained and evaluated on one internal dataset. Generalization to other sign language corpora has not been validated.
