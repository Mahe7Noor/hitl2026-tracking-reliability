# Human-in-the-Loop Active Learning for Reliable Person Detection and Tracking in Overhead Fisheye Videos

**HSBI · HITL Machine Learning · 2026**  
**Team:** [Add team members]  
**Deliverable 1:** [Main Active Learning notebook](active_learning/01_active_learning_tracking_reliability.ipynb) · [Data audit](active_learning/00_data_audit.ipynb)

## Project overview

Our broader project explores how human feedback and Active Learning can support person detection and tracking in overhead fisheye videos while reducing annotation effort. **Deliverable 1** compares four sampling strategies for training a classifier to flag *ground-truth-unmatched person detections*. Identity preservation is the longer-term motivation, **not a measured outcome of this deliverable**.

**Research question:** Under a fixed annotation budget, how do uncertainty, diversity, and representative sampling compare with random sampling in identifying unmatched person detections?

## Pipeline and labels

1. A fine-tuned **HABBOF YOLO-OBB** model detects people using rotated bounding boxes.
2. **ByteTrack** associates predictions across frames and supplies temporal information (track age, motion and gaps).
3. We extract **11 features** from each tracked detection and train a **Logistic Regression** reliability classifier.
4. Predicted boxes are matched to WEPDTOF ground-truth boxes from the same frame using **oriented IoU ≥ 0.50** and **one-to-one Hungarian assignment**: `1 = matched`, `0 = unmatched`.
5. The **simulated annotation oracle** reveals those existing reference labels only when an Active Learning policy selects a sample.

The classifier evaluates **box matching, not the correctness of track IDs**. Unmatched predictions may be duplicates or poorly localized boxes; completely missed people are not represented in its input rows.

## Experimental setup

| Setting | Value |
|---|---|
| Data | WEPDTOF Repair Store: 947 frames, 13,123 tracked detections |
| Partitions | Train: frames 1–620; excluded: 621–699; test: 700–947 |
| Classifier | Class-weighted Logistic Regression, median imputation and standardization |
| Strategies | Random, Uncertainty, Diversity, Representative |
| Label budget | 20 initial labels (10 per class) + 20 per round × 6 rounds = **140** |
| Repetitions | **5 seeds** (11, 22, 33, 44, 55) |
| Evaluation | Unmatched-class Average Precision (AP); supplementary F1 |

All four policies use the same initial labeled examples per seed, model, labeling budget and held-out test set. Learning-curve lines show five-seed means; shading denotes **±1 sample standard deviation**.

### Final results — full 947-frame experiment

| Strategy | Mean final AP | SD |
|---|---:|---:|
| **Representative** | **0.9885** | 0.0032 |
| Random | 0.9852 | 0.0020 |
| Uncertainty | 0.9836 | 0.0068 |
| Diversity | 0.9644 | 0.0115 |

Representative has a **small** final advantage over random (~0.33 percentage points); uncertainty is similar to random, while diversity performs worse. The [main notebook](active_learning/01_active_learning_tracking_reliability.ipynb) contains learning curves, an optional confidence-only reference, and detailed discussion. These results do **not** demonstrate improved YOLO detections or ByteTrack identities.

## Repository contents

```text
active_learning/   # Main experiment notebook and data audit
src/reliability_al/ # Feature extraction, labeling, classifier and policies
scripts/           # Build labels, run experiments, audit, submission checks
tests/             # Automated tests
results/full/      # Saved full experiment figures and summary results
requirements.txt   # Dependencies
```

Large WEPDTOF images/annotations, model weights and videos are not included. Their sources and expected locations are described below.

## Reproduce on HSBI yourAI

Requires Python 3.9+ with dependencies from `requirements.txt`. With the WEPDTOF Repair Store annotations and previously generated YOLO + ByteTrack outputs under the same `Wepdtof/` parent directory:

```bash
cd /home/mnoor1/person_tracking/Wepdtof/hitl2026-tracking-reliability
source ../.venv_yolo26/bin/activate
# If needed: python -m pip install -r requirements.txt

# Only if data/real/repair_store_reliability.csv.gz is missing:
python scripts/build_reliability_labels.py \
  --tracks ../runs/track/habbof_finetuned_repair_store_v2/tracks.csv \
  --annotations ../Wepdtof_raws/repair_store/annotations \
  --mapping ../runs/track/habbof_finetuned_repair_store_v2/frame_mapping.csv \
  --scene repair_store --require-all-frames \
  --output data/real/repair_store_reliability.csv.gz

# Only if full experiment results are missing:
python scripts/run_experiments.py \
  --dataset data/real/repair_store_reliability.csv.gz \
  --out results/full --rounds 6 --batch-size 20

python -m pytest -q
python scripts/check_deliverable.py
```

Open both notebooks in JupyterLab, select the project Python kernel, run all cells, confirm **`DATA MODE: FULL`**, and save the outputs before committing. The original data and tracker outputs must be obtained or generated separately; the repository does not download them automatically.

## Limitations and next steps

Results come from **one video scene** with a temporal train/test split. Labels depend on the IoU threshold and matching quality; AP is not tracking identity accuracy. The simulated oracle is not a real human study. Future work may introduce human box corrections (e.g. through CVAT), retrain the reliability classifier, and assess whether corrected annotations can improve YOLO and downstream ByteTrack performance. Those improvements have **not yet been demonstrated**.


