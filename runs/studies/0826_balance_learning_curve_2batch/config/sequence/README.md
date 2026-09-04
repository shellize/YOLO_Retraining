# Balance Test two-batch learning curve

This study keeps the Balance Test validation and test batches fixed and uses
the remaining 16 physical batches as eight cumulative Full Cold stages.

- `ordered.yaml`: chronological training-batch order.
- `random_s41.yaml`: batch permutation generated with seed 41.
- `random_s43.yaml`: batch permutation generated with seed 43.

All three sequences use the same Sequence seed 42, which gives matching Task
seeds 42 through 49 at the corresponding stages. The permutation seed changes
only which two physical batches arrive at each stage. Every Task cold-starts
from `yolov5s.pt`, trains for 100 epochs, selects `best.pt` by validation mAP50,
and evaluates on the same Balance Test test split. The final report records both
per-class AP50 and per-class AP50-95; the per-class plot uses AP50 so its
threshold matches the primary overall learning curve.

Run all three serially with:

```bash
bash runs/studies/0826_balance_learning_curve_2batch/scripts/run_balance_learning_curve.sh
```

To wait until physical GPU 0 has remained idle for ten continuous minutes and
then launch automatically:

```bash
bash runs/studies/0826_balance_learning_curve_2batch/scripts/run_balance_learning_curve_when_gpu_idle.sh
```

The launcher refuses to overwrite incomplete output and skips completed
sequences. The report generator and its combined CSV/plots live together under
`charts/balance_learning_curve_2batch/`.
