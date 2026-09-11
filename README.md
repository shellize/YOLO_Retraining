# YOLO Retraining

Research-oriented sequential retraining experiments. The default detector is the original, anchor-based YOLOv5s from Ultralytics YOLOv5 v7.0 at commit `915bbf294bb74c859f0b41f1c23bc395014ea679`. The newer `ultralytics` package remains an optional backend rather than a default dependency.

The framework separates one independent `Task` from a `SequenceExperiment`, which invokes complete Tasks in arrival order and passes the previous result directory explicitly.

## Phase 1 baselines

| Baseline | Initialization | Training data |
|---|---|---|
| Full Cold | pretrained | all seen groups |
| Full Warm | previous Task `best.pt` by default | all seen groups |
| Current-only | previous Task `best.pt` by default | current group |
| Random Replay | previous Task `best.pt` by default | current + fixed random history subset |

VPS, AFSS, resume, COCO input, and dynamic epoch sampling are deliberately out of scope for phase 1.

## Environment

Create the isolated Python 3.10 environment, install PyTorch 2.2.2/CUDA 12.1 dependencies, clone the fixed YOLOv5 source revision, and download `yolov5s.pt`:

```powershell
.\requirements\bootstrap.ps1
```

```bash
./requirements/bootstrap.sh
```

The default environment is `yolo-retraining-v5`. YOLOv5 is stored in the ignored `.third_party/yolov5` directory. Set `YOLOV5_ROOT` before running a script to use an existing server checkout. The checkout must be clean and exactly match the fixed commit; scripts never reset or overwrite a mismatched source tree.

The large CUDA wheels are downloaded into ignored `.third_party/wheels` files with retry and resume support. A server mirror can be selected with `YOLO_RETRAINING_TORCH_WHEEL_URL` and `YOLO_RETRAINING_TORCHVISION_WHEEL_URL`; the default URLs remain the official PyTorch CUDA 12.1 index.

If the environment and source already exist, use `requirements/use_existing_env.ps1` or `requirements/use_existing_env.sh`. Bootstrap refuses to mutate an existing environment unless explicitly allowed with `-AllowExistingEnvironmentUpdate` on Windows or `YOLO_RETRAINING_ALLOW_ENV_UPDATE=1` on Linux.

Environment setup scripts live under `requirements/`; experiment launchers live under `scripts/`. To run the formal Full Cold base task with pretrained YOLOv5s weights and all of `stage0` on the second physical GPU:

```bash
bash scripts/run_full_cold_stage0.sh
```

The selected physical GPU is exposed as logical `device=0` inside the training process. Experiment parameters such as seed, epochs, batch size, and output location are defined by the YAML configuration.

Run the two baseline lanes in separate terminals. Each GPU runs one heavy and one light Sequence serially; the two terminals run in parallel:

```bash
# Terminal 1, physical GPU 0
bash scripts/run_full_cold_current_only.sh

# Terminal 2, physical GPU 1
bash scripts/run_full_warm_random_replay.sh
```

The allocation is fixed: GPU 0 runs Full Cold then Current-only; GPU 1 runs Full Warm then Random Replay. The launcher scripts write these physical GPU IDs directly.

The optional modern Ultralytics backend can be installed separately:

```powershell
conda run -n yolo-retraining-v5 python -m pip install -r requirements/ultralytics.lock
```

## Data

The default data protocol is frozen by explicit UTF-8 TXT manifests. [`configs/data/self_improving.yaml`](configs/data/self_improving.yaml) maps `stage0` through `stage7`, fixed `val`, complete `test`, and the nested `test_filtered`/`test_difficult` subsets to the checked-in manifests under `configs/data/manifests/self_improving/`:

```yaml
groups:
  test:
    split: test
    manifest: ../../configs/data/manifests/self_improving/test.txt
  test_filtered:
    split: test
    subset_of: test
    manifest: ../../configs/data/manifests/self_improving/test_filtered.txt
  stage0:
    split: train
    manifest: ../../configs/data/manifests/self_improving/stage0.txt
```

The standard Task and Sequence configs evaluate both `test` and `test_filtered`. To also report the excluded 55-image difficult subset, append `test_difficult` to `data.test`, for example `--set 'data.test=[test,test_filtered,test_difficult]'`. A `subset_of` relation explicitly permits a test manifest to overlap its parent while still rejecting accidental overlap between train, validation, or unrelated test groups.

The older directory-based layout and catalog protocols remain supported only for historical compatibility; the current default protocol is manifest-only. Images must use the native `images/...` and `labels/...` layout. Algorithms operate stable IDs of the form:

```text
group_id::relative/path/to/image.jpg
```

Selected images are passed to the detector through a small text manifest. Images and labels are never copied, moved, hard-linked, or soft-linked.

A manifest contains one image path per non-comment line, relative to the TXT
file. Absolute paths remain accepted but are less portable:

```yaml
groups:
  stage0:
    split: train
    manifest: ../../data_analyse/custom_dataset/results/manual/stage0.txt
```

Use `data_analyse/custom_dataset/custom_dataset.py` to write manifests, build
or validate layouts, and create a deliberate random-frame diagnostic split
while preserving the existing group names and exact budgets. Use
`data_analyse/dataset_redundancy/redundancy_analysis.py` for the independent
YOLOv5s-feature redundancy audit and matched-size random controls.

Existing datasets named `0720_01` through `0720_09` can be normalized once on each machine. The command renames both image and label directories and rewrites the dataset-local `data.yaml` with `path: .` and the fixed train/val/test split:

```bash
conda run -n yolo-retraining-v5 python scripts/normalize_batch_names.py /data/yolo/self_improving
```

## Run

```powershell
yolo-retraining task --config configs/task/full_cold.yaml --set backend.params.device=0
```

Warm-start Task configs require a parent result directory:

```powershell
yolo-retraining task --config configs/task/full_warm.yaml --set task.parent_result='runs/tasks/previous-task'
```

Sequence:

```powershell
yolo-retraining sequence --config configs/sequence/random_replay.yaml
```

For sequence experiments, `initialization.subsequent.checkpoint` controls the
parent checkpoint and defaults to `best`. Use an override such as
`--set initialization.subsequent.checkpoint=last` when the final epoch
checkpoint is the intended baseline. This is a weight-only warm start, not an
optimizer or scheduler resume.

The checked-in Task and Sequence configs use the logical layout. Sequence arrivals list only stage IDs; the common layout supplies their physical folders and the fixed validation/test groups.

### B640 selection study

The ten configs under `configs/sequence/selection_study` compare `positive_only`, `pred_positive`, `error_hard`, `gradnorm_topk`, and the equal-budget `random_topk` control under cumulative-cold and current-warm updates. They use the Balance Test layout, `imgsz=640`, and GPU device 0. Every sequence bootstraps from the completed Balance Test Full Warm Stage 0 task, so Stage 0 is not retrained. Teacher-based selection also reuses that completed Full Warm lineage's Stage 0/1/2 `best.pt` checkpoints at confidence 0.25 and IoU 0.5.

On the server, launch the whole study only after physical GPU0 has been observed idle for ten continuous minutes:

```bash
bash scripts/run_selection_study_when_gpu0_idle.sh
```

The waiter samples GPU state every 60 seconds, resets the ten-minute timer after any busy sample, rejects concurrent waiters through a lock, runs a full config/checkpoint preflight, and then executes all ten sequences serially. Completed sequences are skipped; an incomplete pre-existing output stops the launch for manual inspection.

## Results

Every completed Task contains resolved `task.yaml`, selected ID files, `last.pt`, `best.pt`, train/evaluation metrics, cost accounting, and `task_result.json`. Evaluation reports AP@0.1, AP@0.2, AP@0.3, AP@0.5, and standard AP@0.5:0.95; AP@0.3 is the default primary metric. By default, each `best` evaluation for `test`, `test_filtered`, and the optionally enabled `test_difficult` group also writes `predictions.jsonl` and `confidence_sweep.json` next to `metrics.json`. These artifacts are generated from the same inference pass. A failed Task writes `task_status.json` and `logs/error.txt`, does not write `task_result.json`, and stops its Sequence. Resume and overwrite are not supported.

`predictions.jsonl` has one line per evaluated image. It stores normalized ground-truth and predicted boxes, prediction confidence/class, and correctness at IoU thresholds `[0.1, 0.2, 0.3, 0.5, ..., 0.95]`, so later subset analysis does not require another inference pass. `confidence_sweep.json` uses IoU 0.3 by default and records the seven confidence thresholds `0.2` through `0.8`, including macro/micro Precision, Recall, F1, TP, FP, and FN, plus per-class curves. To change the artifact scope or confidence list, set `evaluation.prediction_artifact_checkpoints`, `evaluation.prediction_artifact_groups`, or `evaluation.confidence_sweep_thresholds`; set `evaluation.save_prediction_artifacts: false` to disable them.

Each completed Task also writes local TensorBoard events under its `tensorboard/` directory. To compare all Tasks and Sequence Tasks under the project `runs` directory, start TensorBoard from the project root:

```powershell
conda run -n yolo-retraining-v5 tensorboard --logdir .\runs --port 6006
```

Open `http://localhost:6006`. The event files contain training history, evaluation metrics, cost metrics, and the resolved experiment configuration. No Comet/ClearML account or upload is involved. Existing runs can be viewed only when their event files already exist; `results.csv` alone is not automatically imported by TensorBoard.

YOLOv5 subprocess logs are compact by default: progress bars keep their final state per scan/epoch/evaluation, and repeated read-only incomplete-JPEG warnings are written as a count with a few examples. The terminal keeps coarse epoch and validation progress. Set `YOLO_RETRAINING_LOG_MODE=full` when the complete subprocess stream is needed for debugging.

## Tests

```powershell
conda run -n yolo-retraining-v5 python -m pytest -q
```

Default tests use a fake backend and require no network or GPU:

```powershell
conda run -n yolo-retraining-v5 python -m pytest -q
```

The original YOLOv5s real-data smoke deterministically selects `64 train / 16 val / 16 test` samples from `YOLO_RETRAINING_REAL_DATA_ROOT`, `data/organized`, or the current local fallback `data/self_improving` (in that priority order), and trains one epoch:

```powershell
$env:YOLO_RETRAINING_RUN_V5_SMOKE="1"
conda run -n yolo-retraining-v5 python -m pytest -q tests/test_yolov5_smoke.py
```

This sample crosses the original batches only to test the engineering pipeline; it is not a paper experiment. Optional Ultralytics tests use `YOLO_RETRAINING_RUN_ULTRALYTICS_SMOKE=1`. Two-GPU server tests use `YOLO_RETRAINING_RUN_DDP_SMOKE=1` and `device=[0,1]`.

## YOLOv5 semantics

Training calls the original `train.py` in an isolated subprocess. Its SGD, augmentation, AutoAnchor, and AMP checks remain intact. The adapter extends validation with AP@0.1/AP@0.2/AP@0.3 while reconstructing the unchanged standard AP@0.5 and AP@0.5:0.95 values. By default, `best.pt` and early stopping use validation AP@0.3. Set `backend.params.best_metric: map50` for AP@0.5 selection or `backend.params.best_metric: yolov5_fitness` for the original `0.1 × AP@0.5 + 0.9 × AP@0.5:0.95` rule. The effective rule is recorded in Task provenance, and per-epoch low-IoU validation metrics are saved in `metrics/validation_iou_metrics.jsonl` and merged into `metrics/train_history.csv`. A later Sequence Task initializes from the previous `best.pt` by default, or from the checkpoint selected by `initialization.subsequent.checkpoint`; this is a weight-only warm start, not an optimizer/scheduler resume.

The subprocess invokes the fixed source's original `train.main()` through a narrow adapter. YOLOv5 v7.0 normally rewrites JPEG files whose end marker is incomplete; the adapter suppresses only that write-back and accepts the image read-only. This preserves immutable source data without copying images or modifying the pinned YOLOv5 checkout, and the adaptation is recorded in `task_result.json` provenance.
