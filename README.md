# rehab-pipeline

Skeleton-based rehab exercise classification: collects exercise-demonstration
video, extracts skeleton keypoints, and trains a CTR-GCN model to classify
rehab exercises for a given body region ("domain"), with a real-time
inference path and a multi-domain stroke-assessment orchestrator on top.
Packaged as an installable Python package (`rehab_pipeline`) with a set of
`rehab-*` console commands, one per pipeline stage.

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
| `stroke_rehab` | Orchestration layer, not a domain | `rehab-stroke-assessment` runs `upper_body` + `lower_limb` (and `face` once it exists) in sequence and combines them into one report -- stroke assessment isn't its own body region |

Both implemented domains' exercise lists are a starting point, not a
clinician-validated taxonomy -- see Known limitations.

## Architecture

```
pyproject.toml              Package metadata, dependencies, rehab-* console script entry points
requirements.txt            Fallback for plain `pip install -r` (pyproject.toml is primary)

src/rehab_pipeline/
├── domains/                 Per-domain config: classes, search queries, joint indices,
│   │                        skeleton graph, normalization reference
│   ├── base.py               DomainConfig dataclass + shape/consistency checks
│   ├── lower_limb.py
│   └── upper_body.py
│
├── common/                  Shared logic used by every stage -- this is what makes
│   │                        the pipeline domain-agnostic instead of copy-pasted per domain
│   ├── preprocessing.py      Interpolation, center/scale normalization, tensor building
│   ├── crypto.py              At-rest encryption for skeleton tensors (Fernet)
│   ├── quality_filter.py     Automated YOLO triage before a human ever reviews a clip
│   └── inference.py           Model loading + classification, shared by realtime + stroke assessment
│
├── pipeline/                 The linear per-domain pipeline, one module per stage
│   ├── collect.py             Stage 1: download candidates -> pending_review/      [rehab-collect]
│   ├── review_app.py          Stage 2: human labeling UI -> raw/ (confirmed only)  [rehab-review]
│   ├── extract.py             Stage 3: skeleton extraction -> encrypted skeletons/ [rehab-extract]
│   ├── augment.py             Optional: fills classes via flip/brightness aug      [rehab-augment]
│   ├── split.py                Stage 4: subject-grouped train/val/test split        [rehab-split]
│   ├── train.py                Stage 5: train CTR-GCN + calibrate confidence         [rehab-train]
│   └── export.py               Stage 6: export ONNX + deployment metadata            [rehab-export]
│
├── serve/                    Runtime inference, built on common/inference.py
│   ├── realtime.py            Stage 7: live webcam inference, single domain    [rehab-realtime]
│   └── stroke_assessment.py   Multi-domain orchestrator ("stroke_rehab")       [rehab-stroke-assessment]
│
├── maintenance/
│   └── purge_rejected.py     Retention utility for unreviewed clips            [rehab-purge]
│
└── run_pipeline.py           Convenience runner for stages 1/3/4 of one domain [rehab-run-pipeline]

datasets/<domain>/            Data, not code -- pending_review/, raw/, skeletons/, *.csv, *.json
models/<domain>/               best_model.pth, best_model.onnx, deployment_metadata.json
evaluation/<domain>/           metrics.json, confusion_matrix.png/.json, calibration.json
```

Each pipeline/serve/maintenance module exposes both a `main()` (used by its
console script) and its underlying function (`download_and_process`,
`process_dataset`, `split_dataset`, etc.) so other code -- like
`run_pipeline.py` or `stroke_assessment.py` calling `serve/realtime.py`'s
`run()` directly -- can call it in-process instead of shelling out.

Nothing else should be in this repo -- earlier one-off patch scripts tied to
a since-replaced data schema (`fix_metadata.py`, `repair_metadata.py`,
`force_fill.py`, `cleanup_skeletons.py`, `audit_pipeline.py`,
`collect_dataset.py`, `clean_dataset.py`, `test_mp.py`) have been removed;
anything useful in them (3-frame duplicate hashing, the YOLO visibility
heuristic) was folded into the modules above.

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate

# GPU (CUDA) users only -- install this BEFORE the editable install below,
# so pip doesn't pull a CPU-only torch wheel from PyPI instead:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Installs the package + registers every rehab-* command on your PATH
pip install -e .

# MediaPipe pose model asset (not committed to git -- it's a downloadable
# third-party asset, not project source). Run from the repo root:
curl -L -o pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
# If that URL has moved, get the current one from
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

rehab-keygen   # generates .keys/skeleton.key -- back this up, never commit it
```

`pip install -e .` is the recommended path (it gives you the `rehab-*`
commands used throughout this README). If you'd rather not install the
package, every command below also works as `python -m rehab_pipeline.<module>`
run from the repo root, e.g. `python -m rehab_pipeline.pipeline.collect
--domain lower_limb --target 50`.

**All commands assume you're running from the repository root** -- paths
like `datasets/<domain>/` and `pose_landmarker_heavy.task` are resolved
relative to the current directory, not installed into site-packages.

## Running the pipeline (per domain)

Every stage takes `--domain lower_limb` or `--domain upper_body`. Run the
whole sequence once per domain you're building.

### 1-2. Collect + review
```bash
rehab-collect --domain lower_limb --target 50   # -> pending_review/
rehab-review --domain lower_limb                # http://127.0.0.1:5050
```
To review two domains at once, give the second instance its own port:
`rehab-review --domain upper_body --port 5051`.

`rehab-review` serves raw video over local HTTP with no auth -- run it on
localhost only, never expose the port. Rejected clips are deleted
immediately; nothing rejected is retained. Confirmed clips move to `raw/`
with a full audit trail in `datasets/<domain>/review_log.csv` (reviewer,
timestamp, original query-based guess vs. confirmed label).

### 3. Extract skeletons
```bash
rehab-extract --domain lower_limb
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
rehab-augment --domain lower_limb --target 50
```
Generates flip/brightness variants to top up under-represented classes.
Run this *before* extraction (step 3) so augmented clips get skeletons too.
`rehab-split` groups every augmented clip with its source video so these
never leak across train/val/test.

### 4. Split
```bash
rehab-split --domain lower_limb
```
Produces `train_labels.csv` / `val_labels.csv` / `test_labels.csv` (70/15/15),
grouped by source video.

### 5. Train
```bash
rehab-train --domain lower_limb
```
Uses the validation set (not the test set) for early stopping and
checkpoint selection. After training, fits a calibration temperature and a
recommended confidence threshold on the validation set, then evaluates the
test set exactly once. Outputs land in `evaluation/<domain>/`: `metrics.json`,
`confusion_matrix.png`/`.json`, `calibration.json`.

### 6. Export
```bash
rehab-export --domain lower_limb
```
Exports `models/<domain>/best_model.onnx` + `deployment_metadata.json`,
which includes the calibration temperature and confidence threshold
`rehab-realtime` needs.

### 7. Real-time deployment (single domain)
```bash
rehab-realtime --domain lower_limb --source 0 --window-seconds 6
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
rehab-stroke-assessment --domains upper_body lower_limb
```
Prompts you to perform each domain's movement in turn, captures one window
per domain via the same logic as `rehab-realtime`, and prints/saves a
combined JSON report. This is the "stroke_rehab" use case: an orchestration
layer over the per-domain models, not a separate model or class list. It
carries an explicit disclaimer in its own output -- it's a screening report,
not a diagnosis. Add `face` to `--domains` once a `face` domain exists;
until then it's reported as `not_available`.

### Convenience: stages 1/3/4 in one command
```bash
rehab-run-pipeline --domain lower_limb --target 50
```
Runs collect, then stops and tells you to run `rehab-review` if there's
anything pending, then (once the queue is empty) runs extract + split.
Training/export are deliberately separate commands -- they're typically run
on a GPU machine.

## Adding a new domain

1. Create `src/rehab_pipeline/domains/<name>.py` following `upper_body.py` as
   a template: `CLASSES`, `QUERIES` (search phrases per class), MediaPipe
   joint indices + names, `center_joints`/`scale_joints` (pick a pair that
   stays roughly rigid across the domain's movements -- never the pair
   whose distance IS the motion being classified), `graph_edges` (skeleton
   connectivity), `yolo_pose_keypoint_range` (COCO keypoint slice for the
   collection-time visibility triage).
2. Register it in `src/rehab_pipeline/domains/__init__.py`.
3. Everything else (collection, review, extraction, split, train, export,
   realtime inference) works automatically via `--domain <name>` -- none of
   those modules have per-domain logic in them anymore.
4. **Face is the one domain that won't fit this pattern.** It needs
   MediaPipe FaceLandmarker (468 points, not Pose's 33) and, if the goal is
   droop/symmetry assessment, a fundamentally different output (a symmetry
   score/regression) rather than a 9-way softmax classifier. Don't force it
   through the `DomainConfig` shape above without rethinking
   `common/preprocessing.py`'s classification-specific assumptions first.

## Data handling / privacy posture

- **Raw video is never retained** past the review step (rejected) or the
  extraction step (accepted). Only the derived skeleton keypoint tensor is
  kept, and it's encrypted at rest (`common/crypto.py`, Fernet).
- The encryption key (`.keys/skeleton.key`) is dev-grade key management: a
  local file, gitignored. **Before handling real patient data in
  production, replace this with a real KMS** (AWS/GCP/Azure) and key
  rotation -- this deliberately does not pretend otherwise.
- Every accepted training label has a human review record
  (`datasets/<domain>/review_log.csv`).
- `rehab-purge --domain <name>` cleans up unreviewed clips that sat in the
  queue past a retention window (default 30 days, dry-run by default).
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
- `rehab-stroke-assessment` is a thin sequencing layer, not a validated
  composite stroke score. It reports each domain's independent result; it
  does not combine them into any kind of clinical severity scale.
- Labels are human-confirmed, but by whoever runs `rehab-review` -- they
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
