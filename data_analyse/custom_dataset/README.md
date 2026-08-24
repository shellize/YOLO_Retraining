# Custom TXT datasets

This tool creates flexible train/validation/test layouts without copying images.
Manifest paths are resolved relative to the TXT file. The layout's `manifest`
path is resolved relative to the dataset root.

Create the deliberately easy frame-random diagnostic split while preserving
the original `test`, `val`, and `stage0`-`stage3` names and exact budgets:

```powershell
conda run -n yolo-retraining-v5 python data_analyse/custom_dataset/custom_dataset.py random-reassign `
  --layout configs/data/self_improving.yaml `
  --output-dir data_analyse/custom_dataset/results/random_frame_s42 `
  --seed 42
```

Validate any generated layout:

```powershell
conda run -n yolo-retraining-v5 python data_analyse/custom_dataset/custom_dataset.py validate `
  --layout data_analyse/custom_dataset/results/random_frame_s42/layout.yaml
```

Build a layout from arbitrary manifests:

```powershell
conda run -n yolo-retraining-v5 python data_analyse/custom_dataset/custom_dataset.py build-layout `
  --dataset-root data/self_improving `
  --names "large luggage,stroller,wheelchair,flatbed truck" `
  --manifest "stage0:train:E:/manifests/stage0.txt" `
  --manifest "val:val:E:/manifests/val.txt" `
  --manifest "test:test:E:/manifests/test.txt" `
  --output data_analyse/custom_dataset/results/manual/layout.yaml
```

The resulting layout can be passed directly to the existing runner with
`--set data.layout=<layout.yaml>`. Raw images and labels are never modified.

Create a manifest from one or more physical image directories:

```powershell
conda run -n yolo-retraining-v5 python data_analyse/custom_dataset/custom_dataset.py write-manifest `
  --dataset-root data/self_improving `
  --images images/0720_14 `
  --output data_analyse/custom_dataset/results/one_batch/0720_14.txt
```
