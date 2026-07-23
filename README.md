# Skeleton-Based Rehab Exercise Classification Pipeline

Collects exercise-demonstration video, extracts skeleton keypoints, and
trains a CTR-GCN model to classify rehab exercises for a given body region
("domain"), with a real-time inference path and a multi-domain stroke
assessment orchestrator on top.

**Read [Known limitations](#known-limitations-read-before-clinical-use) before
treating any model trained here as patient-ready.** This pipeline fixes the
engineering gaps that made the original version untrustworthy (unverified
labels, train/test leakage, no confidence gating, raw video retention). It
does **not** by itself make a model clinically validated -- that requires
validating error rates against your actual target patient population and,
likely, regulatory input.

## Scope

| Domain | Status | Classes |
|---|---|---|
| `lower_limb` | Implemented | ankle, calf, hamstring, heel_slide, hip, knee, leg_raise, quadriceps, toes |
| `upper_body` | Implemented | shoulder_flexion, shoulder_abduction, elbow_flexion, wrist_curl, pronation_supination, arm_raise, external_rotation, bicep_curl, tricep_extension |
| `face` | **Not built** | Facial droop/symmetry assessment needs a different model (MediaPipe FaceLandmarker + a symmetry score, not this repo's CTR-GCN classifier) -- see [Known limitations](#known-limitations-read-before-clinical-use) |
| `stroke_rehab` | Orchestration layer, not a domain | `stroke_assessment.py` runs `upper_body` + `lower_limb` (and `face` once it exists) in sequence and combines them into one report -- stroke assessment isn't its own body region |

Both implemented domains' exercise lists are a starting point, not a
clinician-validated taxonomy -- see Known limitations.

## Repo layout

```
domains/                  Per-domain config: classes, search queries, joint indices,
                           skeleton graph, normalization reference (see domains/base.py)
  lower_limb.py
  upper_body.py

pipeline_common.py         Shared preprocessing (interpolation, normalization, tensor
                           building) -- takes a domain config, used by every stage
quality_filter.py          Automated YOLO triage before a human ever reviews a clip
crypto_utils.py             At-rest encryption for skeleton tensors
inference_common.py         Shared model-loading + classification logic for realtime_infer.py
                           and stroke_assessment.py (keeps their preprocessing identical)

phase_1_2_collect.py       Stage 1: download candidates -> pending_review/
review_app.py               Stage 2: human labeling UI -> raw/ (confirmed only)
phase_3_4_extract.py       Stage 3: skeleton extraction -> encrypted skeletons/
augment_fill.py             Optional: fills classes up to target via flip/brightness augmentation
phase_5_split.py           Stage 4: subject-grouped train/val/test split
phase_6_7_train.py         Stage 5: train CTR-GCN + calibrate confidence
phase_8_9_export.py        Stage 6: export ONNX + deployment metadata
realtime_infer.py           Stage 7: live webcam inference, single domain
stroke_assessment.py        Multi-domain orchestrator (the "stroke_rehab" use case)

purge_rejected.py           Retention utility for unreviewed clips
run_pipeline.py             Convenience runner for stages 1, 3, 4 of one domain
```

Nothing else should be in this repo -- earlier one-off patch scripts tied to
a since-replaced data schema (`fix_metadata.py`, `repair_metadata.py`,
`force_fill.py`, `cleanup_skeletons.py`, `audit_pipeline.py`,
`collect_dataset.py`, `clean_dataset.py`, `test_mp.py`) have been removed;
anything useful in them (3-frame duplicate hashing, the YOLO visibility
heuristic) was folded into the scripts above.

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or CPU wheel
pip install -r requirements.txt

# MediaPipe pose model asset (not committed to git -- it's a downloadable
# third-party asset, not project source):
curl -L -o pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
# If that URL has moved, get the current one from
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

python crypto_utils.py init   # generates .keys/skeleton.key -- back this up, never commit it
```

## Running the pipeline (per domain)

Every stage takes `--domain lower_limb` or `--domain upper_body`. Run the
whole sequence once per domain you're building.

### 1-2. Collect + review
```bash
python phase_1_2_collect.py --domain lower_limb --target 50   # -> pending_review/
python review_app.py --domain lower_limb                      # http://127.0.0.1:5050
```
To review two domains at once, give the second instance its own port:
`python review_app.py --domain upper_body --port 5051`.

`review_app.py` serves raw video over local HTTP with no auth -- run it on
localhost only, never expose the port. Rejected clips are deleted
immediately; nothing rejected is retained. Confirmed clips move to `raw/`
with a full audit trail in `datasets/<domain>/review_log.csv` (reviewer,
timestamp, original query-based guess vs. confirmed label).

### 3. Extract skeletons
```bash
python phase_3_4_extract.py --domain lower_limb
```
Only reads from `raw/` (i.e. only human-confirmed videos). Normalizes each
skeleton per the domain's config (e.g. hip-centered/hip-width-scaled for
lower_limb, shoulder-centered/shoulder-width-scaled for upper_body) so the
model learns movement, not camera framing. Encrypts the resulting tensor to
`skeletons/*.npy.enc`, then **deletes the source video** -- raw video is
never retained past this step. Clips with poor visibility of the relevant
joints are dropped rather than silently trained on (see
`validation_report.json`).

### (Optional) Fill classes via augmentation
```bash
python augment_fill.py --domain lower_limb --target 50
```
Generates flip/brightness variants to top up under-represented classes.
Run this *before* extraction (step 3) so augmented clips get skeletons too.
`phase_5_split.py` groups every augmented clip with its source video so
these never leak across train/val/test.

### 4. Split
```bash
python phase_5_split.py --domain lower_limb
```
Produces `train_labels.csv` / `val_labels.csv` / `test_labels.csv` (70/15/15),
grouped by source video.

### 5. Train
```bash
python phase_6_7_train.py --domain lower_limb
```
Uses the validation set (not the test set) for early stopping and
checkpoint selection. After training, fits a calibration temperature and a
recommended confidence threshold on the validation set, then evaluates the
test set exactly once. Outputs land in `evaluation/<domain>/`: `metrics.json`,
`confusion_matrix.png`/`.json`, `calibration.json`.

### 6. Export
```bash
python phase_8_9_export.py --domain lower_limb
```
Exports `models/<domain>/best_model.onnx` + `deployment_metadata.json`,
which includes the calibration temperature and confidence threshold
`realtime_infer.py` needs.

### 7. Real-time deployment (single domain)
```bash
python realtime_infer.py --domain lower_limb --source 0 --window-seconds 6
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

### 8. Combined stroke-rehab assessment (multi-domain)
```bash
python stroke_assessment.py --domains upper_body lower_limb
```
Prompts you to perform each domain's movement in turn, captures one window
per domain via the same logic as `realtime_infer.py`, and prints/saves a
combined JSON report. This is the "stroke_rehab" use case: an orchestration
layer over the per-domain models, not a separate model or class list. It
carries an explicit disclaimer in its own output -- it's a screening report,
not a diagnosis. Add `face` to `--domains` once `domains/face.py` exists;
until then it's reported as `not_available`.

## Adding a new domain

1. Create `domains/<name>.py` following `domains/upper_body.py` as a
   template: `CLASSES`, `QUERIES` (search phrases per class), MediaPipe
   joint indices + names, `center_joints`/`scale_joints` (pick a pair that
   stays roughly rigid across the domain's movements -- never the pair
   whose distance IS the motion being classified), `graph_edges` (skeleton
   connectivity), `yolo_pose_keypoint_range` (COCO keypoint slice for the
   collection-time visibility triage).
2. Register it in `domains/__init__.py`.
3. Everything else (collection, review, extraction, split, train, export,
   realtime inference) works automatically via `--domain <name>` -- none of
   those scripts have per-domain logic in them anymore.
4. **Face is the one domain that won't fit this pattern.** It needs
   MediaPipe FaceLandmarker (468 points, not Pose's 33) and, if the goal is
   droop/symmetry assessment, a fundamentally different output (a symmetry
   score/regression) rather than a 9-way softmax classifier. Don't force it
   through the `DomainConfig` shape above without rethinking `pipeline_common.py`'s
   classification-specific assumptions first.

## Data handling / privacy posture

- **Raw video is never retained** past the review step (rejected) or the
  extraction step (accepted). Only the derived skeleton keypoint tensor is
  kept, and it's encrypted at rest (`crypto_utils.py`, Fernet).
- The encryption key (`.keys/skeleton.key`) is dev-grade key management: a
  local file, gitignored. **Before handling real patient data in
  production, replace this with a real KMS** (AWS/GCP/Azure) and key
  rotation -- this script deliberately does not pretend otherwise.
- Every accepted training label has a human review record
  (`datasets/<domain>/review_log.csv`).
- `purge_rejected.py --domain <name>` cleans up unreviewed clips that sat in
  the queue past a retention window (default 30 days, dry-run by default).
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
- **Neither exercise taxonomy is clinician-validated.** `lower_limb`'s list
  came from the original project; `upper_body`'s list is a reasonable
  starting PT set I proposed, confirmed by the product owner, not a
  clinician. Review both against your actual clinical use case.
- `upper_body`'s `wrist_curl` and `pronation_supination` may not be reliably
  separable from MediaPipe Pose alone -- it gives one positional point per
  wrist, not forearm rotation. Watch the confusion matrix for these two
  once you have real data; you may need MediaPipe Hands landmarks fused in,
  or to merge/drop one of the classes.
- **`face` is not built.** Facial droop/symmetry assessment (the clinically
  standard stroke sign) needs a different landmarker and almost certainly a
  different model type than the classifier used for the limb domains --
  don't bolt it onto the existing `DomainConfig` pattern without rethinking
  that.
- `stroke_assessment.py` is a thin sequencing layer, not a validated
  composite stroke score. It reports each domain's independent result; it
  does not combine them into any kind of clinical severity scale.
- Labels are human-confirmed, but by whoever runs `review_app.py` -- they
  are not clinician-verified unless you have a clinician doing the
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
