# Balance tau=0.90 label-contrast study

This is the same five-experiment study as the previous label-contrast study,
with the adjacent-frame similarity threshold changed from 0.96 to 0.90. The
balanced whole-batch validation/test split is unchanged. Only the four new
Tasks use tau=0.90 manifests; the full-data baseline is the existing completed
Task referenced under `experiment/task/full_raw__existing_result`.

Manifest construction is CPU-only and reuses the cached fixed YOLOv5s
embeddings. Training is forced to physical GPU0 and uses seed 42, YOLOv5s,
640 resolution, batch 64, 300 epochs, and patience 100.

The four new variants are ordinary one-representative deduplication, positive
cluster release with background representatives retained, positive cluster
release with pure-background clusters dropped, and one representative per
cluster with a labeled representative preferred.

Files:

- `scripts/build_manifests.py`: generates the four tau=0.90 variants and links the
  existing baseline.
- `experiment/variants/<variant>/layout.yaml`: complete train/validation/test layout.
- `experiment/variants/<variant>/manifests/train.txt`: merged training manifest.
- `experiment/variants/<variant>/manifests/stage0.txt` through `stage3.txt`: logical stage
  partitions used by the Task config.
- `experiment/variants/<variant>/manifests/val.txt` and `test.txt`: fixed balance holdouts.
- `clusters_tau_0p900.csv`: cluster membership and representative audit.
- `experiment/variants/manifest_summary.json`: generated image/cluster counts and baseline reference.
- `scripts/run_experiments.sh`: serial launcher for the four unfinished Tasks.
- `scripts/run_when_gpu0_idle.sh`: polls GPU0 every three minutes and starts training
  after two consecutive idle checks.

Start the detached GPU0 waiter with:

```bash
tmux new-session -d -s balance_tau090_label_contrast \
  "bash runs/studies/0828_5_dedup_contrast/scripts/run_when_gpu0_idle.sh"
```
