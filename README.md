# Skeleton-Based Lower-Limb Exercise Classification Pipeline

A pipeline that collects exercise-demonstration video, extracts skeleton
keypoints, and trains a CTR-GCN model to classify 9 lower-limb exercises
(ankle, calf, hamstring, heel_slide, hip, knee, leg_raise, quadriceps, toes),
with a real-time inference path for live deployment.

**Read this before treating any model trained here as patient-ready** -- see
[Known limitations](#known-limitations-read-before-clinical-use) at the
bottom. This pipeline fixes the engineering gaps (unverified labels,
train/test leakage, no confidence gating, raw video retention) that made the
previous version untrustworthy. It does **not** by itself make the model
clinically validated -- that requires validating error rates against your
actual target patient population and, likely, regulatory input.

## Pipeline stages

```
1. Collect        phase_1_2_collect.py     -> datasets/lower_limb/pending_review/
2. Review          review_app.py            -> datasets/lower_limb/raw/ (human-confirmed only)
3. Extract         phase_3_4_extract.py     -> datasets/lower_limb/skeletons/*.npy.enc (encrypted)
4. Split           phase_5_split.py         -> train/val/test_labels.csv (grouped by source video)
5. Train           phase_6_7_train.py       -> models/lower_limb/best_model.pth + calibration
6. Export          phase_8_9_export.py      -> best_model.onnx + deployment_metadata.json
7. Deploy (live)   realtime_infer.py        -> live webcam classification with confidence gating
```

Stage 2 is the critical addition: **nothing enters the training set just
because a search query happened to find it.** `phase_1_2_collect.py` only
ever writes to `pending_review/`; a human has to explicitly confirm the
label in `review_app.py` before a clip moves to `raw/` and becomes eligible
for extraction.

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or CPU wheel
pip install -r requirements.txt
python crypto_utils.py init   # generates .keys/skeleton.key -- back this up, never commit it
```

## Running the pipeline

### 1-2. Collect + review
```bash
python phase_1_2_collect.py --target 50   # downloads candidates into pending_review/
python review_app.py                      # open http://127.0.0.1:5050, confirm/correct/reject each clip
```
`review_app.py` serves raw video over local HTTP with no auth -- run it on
localhost only, never expose the port. Rejected clips are deleted
immediately; nothing rejected is retained.

### 3. Extract skeletons
```bash
python phase_3_4_extract.py
```
Only reads from `raw/` (i.e. only human-confirmed videos). Normalizes each
skeleton (hip-centered, hip-width scaled) so the model learns movement, not
camera framing. Encrypts the resulting tensor to `skeletons/*.npy.enc`, then
**deletes the source video** -- raw video is never retained past this step.
Clips with poor leg visibility are dropped rather than silently trained on
(see `validation_report.json`).

### 4. Split
```bash
python phase_5_split.py
```
Produces `train_labels.csv` / `val_labels.csv` / `test_labels.csv` (70/15/15),
grouped by source video so an augmented (flip/brightness) clip and its
source never land in different splits.

### 5. Train
```bash
python phase_6_7_train.py
```
Uses the validation set (not the test set) for early stopping and
checkpoint selection. After training, fits a calibration temperature and a
recommended confidence threshold on the validation set, then evaluates the
test set exactly once. Outputs: `evaluation/lower_limb/metrics.json`,
`confusion_matrix.png`/`.json`, `calibration.json`.

### 6. Export
```bash
python phase_8_9_export.py
```
Exports ONNX + `deployment_metadata.json`, which now includes the
calibration temperature and confidence threshold `realtime_infer.py` needs.

### 7. Real-time deployment
```bash
python realtime_infer.py --source 0 --window-seconds 6
```
Live webcam inference: extracts skeletons per frame, keeps a sliding window,
resamples/normalizes it identically to training, runs ONNX inference every
`--interval` seconds, and reports **"uncertain"** instead of a forced class
whenever calibrated confidence is below the training-derived threshold.
Nothing is written to disk by default. `--window-seconds` should be tuned to
match how long a typical exercise repetition takes in your deployment
context -- check `duration_seconds` in `metadata.csv` for your training
clips as a starting point, since a mismatched window is a real source of
accuracy loss no amount of model tuning fixes.

## Data handling / privacy posture

- **Raw video is never retained** past the review step (rejected) or the
  extraction step (accepted). Only the derived skeleton keypoint tensor is
  kept, and it's encrypted at rest (`crypto_utils.py`, Fernet).
- The encryption key (`.keys/skeleton.key`) is dev-grade key management: a
  local file, gitignored. **Before handling real patient data in
  production, replace this with a real KMS** (AWS/GCP/Azure) and key
  rotation -- this script deliberately does not pretend otherwise.
- Every accepted training label has a human review record
  (`datasets/lower_limb/review_log.csv`: reviewer, timestamp, original
  query-based guess vs. confirmed label).
- `purge_rejected.py` cleans up unreviewed clips that sat in the queue past
  a retention window (default 30 days, dry-run by default).
- Committing encrypted skeleton tensors to a shared/public git repo is a
  privacy decision, not just an engineering one -- they're inert without the
  key, but they are still derived from real people. Decide deliberately
  whether this repo should be private.

## Known limitations (read before clinical use)

- **Source data is still YouTube exercise-demo footage**, not your actual
  patient population. Domain mismatch (able-bodied demonstrators vs. real
  patients with impaired movement, assistive devices, non-ideal home camera
  setups) is the single biggest risk to real-world accuracy and isn't fixed
  by anything in this pipeline. Validate error rates on a sample of your
  actual deployment population before trusting this for anything
  patient-facing.
- Labels are now human-confirmed, but by whoever runs `review_app.py` --
  they are not clinician-verified unless you have a clinician doing the
  reviewing.
- The confidence-gating threshold and calibration are fit on a validation
  set drawn from the same (YouTube-demo) distribution as training -- they
  will likely be miscalibrated on real patients until re-validated there.
- This is exercise-*type* classification only. It says nothing about
  whether the exercise was performed correctly, rep counts, or safety
  (e.g. loss of balance) -- treat it as one input signal, not a complete
  clinical tool.
- If model output will influence anything a patient acts on, scope the
  regulatory picture (e.g. FDA SaMD / EU MDR / HIPAA-GDPR depending on your
  market and how the output is used) before scaling beyond a pilot.
