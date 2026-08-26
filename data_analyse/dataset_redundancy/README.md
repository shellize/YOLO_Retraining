# Dataset redundancy analysis

This workspace tests whether continuous video frames make the nominal training
set much larger than its effective visual diversity. It never trains YOLO,
changes annotations, removes images, or copies the dataset.

Feature extractor:

- the same pinned YOLOv5 v7.0 source used by the experiments;
- the original COCO-pretrained `yolov5s.pt`, never a dataset-trained `best.pt`;
- spatial-pyramid pooled P3/P4/P5 features from layers 17, 20, and 23
  using 4x4, 2x2, and 1x1 grids, so a static camera background cannot erase
  object-position changes as easily;
- concatenation followed by L2 normalization.

Run the complete ordinary-layout audit:

```powershell
conda run --no-capture-output -n yolo-retraining-v5 python `
  data_analyse/dataset_redundancy/redundancy_analysis.py `
  --layout configs/data/self_improving.yaml `
  --output-dir data_analyse/dataset_redundancy/results/self_improving_yolov5s `
  --device 0 `
  --threshold-workers 8 `
  --temporal-window 1
```

If another process owns the GPU, use `--device cpu`, or extract once and reuse:

```powershell
python data_analyse/dataset_redundancy/redundancy_analysis.py `
  --layout configs/data/self_improving.yaml `
  --output-dir data_analyse/dataset_redundancy/results/self_improving_yolov5s `
  --embeddings data_analyse/dataset_redundancy/results/self_improving_yolov5s/embeddings.npz `
  --device cpu
```

Important outputs:

- `summary.json`: effective train size for each audited threshold;
- `temporal_similarity_pairs.csv`: similar frames within the same physical batch;
- `global_knn_pairs.csv`: global top-k neighbors for broader redundancy review;
- `cross_split_nearest.csv`: each validation/test image's closest train frame;
- `audit/top_temporal_pairs.jpg`: visual audit sheet;
- `variants/dedup_tau_*/layout.yaml`: deduplicated training layouts;
- `variants/random_matched_tau_*_s*/layout.yaml`: same-size random controls.

`--threshold-workers` parallelizes independent threshold post-processing with
separate CPU processes. It preserves the threshold grouping algorithm and
output layout; use `1` for serial reproduction. The similarity matrix stages
still follow `--device`.

The threshold sweep is an audit, not permission to select the best threshold by
test mAP. Choose the primary threshold from visual pair review before training.
The primary redundancy grouping compares adjacent filenames only
(`--temporal-window 1`). Global top-k neighbors remain a separate audit and are
not used to remove images, because the same camera returns approximately every
seven frames in the current data.

## Complete serial server study

Run every requested full-cold experiment from the repository root:

```bash
sh scripts/run_fullcold_redundancy_shift_study.sh
```

The script runs, in order:

1. environment and layout validation;
2. one full YOLOv5s embedding pass and three adjacent-frame layouts at cosine
   thresholds 0.970, 0.997, and 0.999;
3. the original full-data full-cold baseline;
4. three deduplicated full-cold sequences (aggressive, middle, conservative);
5. a frame-random train/validation/test reassignment with unchanged group
   budgets, followed by its full-cold sequence;
6. final CSV/JSON comparison including mAP, large luggage AP, image reads, and
   optimizer steps.

Useful server overrides:

```bash
GPU_ID=1 REDUNDANCY_DEVICE=cpu REDUNDANCY_THRESHOLD_WORKERS=8 \
  TRAIN_BATCH=64 EMBED_BATCH=64 WORKERS=16 STUDY_ID=data-study-v1 \
  sh scripts/run_fullcold_redundancy_shift_study.sh
```

Equal-size random subset layouts are always generated. Train them too by adding
`MATCHED_RANDOM_CONTROLS=1`; set `MATCHED_RANDOM_SEEDS="41 42 43"` for three
seeds. Reuse the same `STUDY_ID` after a failure: completed sequences are
skipped, cached embeddings are reused, and incomplete training directories are
never overwritten automatically.
