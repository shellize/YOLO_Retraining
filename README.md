# YOLO Retraining

Research-oriented baseline retraining experiments for Ultralytics YOLOv8 and YOLO11. The framework separates one independent `Task` from a `SequenceExperiment` that invokes multiple Tasks in arrival order.

## Phase 1 baselines

| Baseline | Initialization | Training data |
|---|---|---|
| Full Cold | pretrained | all seen groups |
| Full Warm | previous Task `last.pt` | all seen groups |
| Current-only | previous Task `last.pt` | current group |
| Random Replay | previous Task `last.pt` | current + fixed random history subset |

VPS, AFSS, resume, COCO input, and dynamic epoch sampling are deliberately out of scope for phase 1.

## Environment

Reuse the existing `yolo-cl` Conda environment without changing its dependencies:

```powershell
./scripts/use_existing_env.ps1
```

```bash
./scripts/use_existing_env.sh
```

The scripts validate Python 3.11, PyTorch 2.6.0, torchvision 0.21.0, and Ultralytics 8.4.102, then install this package with `--no-deps --no-build-isolation`. They do not download or upgrade dependencies. Use `bootstrap.ps1` or `bootstrap.sh` only when the environment is missing or an explicit repair is intended.

## Data

Each arrival points to an Ultralytics-compatible YOLO data YAML containing `path`, `train`, `val`, `test`, and `names`. Images must use the native `images/...` and `labels/...` layout. Algorithms operate stable IDs of the form:

```text
group_id::relative/path/to/image.jpg
```

Selected images are passed to Ultralytics through a small text manifest. Images and labels are never copied, moved, hard-linked, or soft-linked.

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

Real paths and arrivals should be provided by copying the example configs. The checked-in examples intentionally point at `datasets/stage0`, which is ignored by Git.

## Results

Every completed Task contains resolved `task.yaml`, selected ID files, `last.pt`, `best.pt`, train/evaluation metrics, cost accounting, and `task_result.json`. A failed Task writes `task_status.json` and `logs/error.txt`, does not write `task_result.json`, and stops its Sequence. Resume and overwrite are not supported.

## Tests

```powershell
conda run -n yolo-cl python -m pytest -q
```

Default tests use a fake backend and require no network or GPU. Real one-epoch smoke tests are opt-in:

```powershell
$env:YOLO_RETRAINING_RUN_SMOKE=1
conda run -n yolo-cl python -m pytest -q -m smoke
```

Two-GPU server smoke tests use `YOLO_RETRAINING_RUN_DDP_SMOKE=1` and require `device=[0,1]`.
