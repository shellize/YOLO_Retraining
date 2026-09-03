# Balance two-batch ordered dedup contrast

This study compares two deduplicated eight-stage Full Cold sequences with the
existing `Balance_Learning_Curve_ordered` baseline. The validation/test batches,
two-batch stage order, model, seed schedule, batch size, image size, and
initialization semantics follow the ordered baseline. The two new sequences and
the existing ordered baseline all use a 100-epoch budget.

The cached fixed YOLOv5 embeddings from `20260824-182608/embeddings.npz` are
reused, but the old manifests and cluster CSVs are not. They were generated from
a different train/validation/test layout. `build_manifests.py` remaps cached
features by physical image path, then rebuilds clusters against
`balance_2batch_ordered.yaml`.

Variants:

- `dedup_tau_0p990`: one representative image per tau=0.99 cluster.
- `dedup_tau_0p960_mixed_release_drop_pure_background`: at tau=0.96, drop every
  pure-background cluster, keep one representative for a pure-labeled cluster,
  and keep every member of a mixed labeled/background cluster.

Each generated `variants/<name>/layout.yaml` contains `stage0` through `stage7`
manifests plus the unchanged Balance validation and test manifests. Cluster
membership, annotation status, and final selection are recorded in
`variants/<name>/clusters.csv`; aggregate counts are in `manifest_summary.json`.

To poll physical GPU0 every ten minutes and launch after two consecutive idle
checks:

```bash
tmux new-session -d -s balance_lc2b_dedup \
  "bash runs/studies/balance_learning_curve_2batch_dedup_contrast/run_when_gpu0_idle.sh"
```

The idle check requires utilization at or below 5%, memory at or below 1024 MiB,
and no process returned by `nvidia-smi --query-compute-apps`. The two sequences
run serially and completed outputs are skipped; incomplete output directories
are never overwritten.
