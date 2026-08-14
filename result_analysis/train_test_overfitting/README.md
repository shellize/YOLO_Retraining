# Train/Test Overfitting Analysis

这是 `result_analysis` 下的一个分析方法目录。`analyze_train_test_overfitting.py` 是可复用脚本；下面的每个带时间戳子目录都是一次独立分析结果，可以保留不同权重、数据快照或参数设置的结果。该方法重点比较训练集与测试集的 Precision、Recall、mAP50 以及逐图 TP/FP/FN，用于观察过拟合程度。

目录结构：

```text
result_analysis/
└── train_test_overfitting/
    ├── analyze_train_test_overfitting.py
    ├── README.md
    └── 20260814__best__stages-1_2_3__conf-0_25/
        ├── comparison.csv
        ├── comparison.md
        ├── summary.csv
        └── ...
```

## 默认分析

不指定 `--run-dir` 时，脚本会自动查找 `runs/sequences` 下最近的三个已完成 sequence：`full-warm`、`random-replay` 和 `full-cold`。默认分析 stage1、stage2、stage3，因此得到 3×3=9 个实验；如需包含 stage0，使用 `--stages 0-3`。

```powershell
conda run --no-capture-output -n yolo-retraining-v5 python `
  "E:/desktop/project/YOLO Retraining/project_v2/result_analysis/train_test_overfitting/analyze_train_test_overfitting.py" `
  --output-dir "E:/desktop/project/YOLO Retraining/project_v2/result_analysis/train_test_overfitting/20260814__best__stages-1_2_3__conf-0_25" `
  --device 0 --batch 16 --workers 4 --half --skip-missing-ids
```

也可以显式指定一个或多个 sequence；`--run-dir` 可以重复使用：

```powershell
python "E:/desktop/project/YOLO Retraining/project_v2/result_analysis/train_test_overfitting/analyze_train_test_overfitting.py" `
  --run-dir "E:/.../runs/sequences/<sequence-1>" `
  --run-dir "E:/.../runs/sequences/<sequence-2>" `
  --run-dir "E:/.../runs/sequences/<sequence-3>" `
  --stages 1,2,3 `
  --checkpoint best.pt `
  --output-dir "E:/.../result_analysis/train_test_overfitting/<analysis-name>"
```

## 关键参数与指标含义

- `--checkpoint best.pt` 是默认值；也可以传 `last.pt`，或传一个具体权重文件路径。每个 task 都会优先在该 task 的 `checkpoints/` 下解析相对文件名。
- `--output-dir` 是结果目录，可选；省略时会在本目录下自动创建带时间戳的子目录。
- `--conf-thres 0.25` 是默认的统一工作点。它同时用于所有实验、训练集和测试集的固定阈值 Precision/Recall 以及逐图 TP/FP/FN；如果测试协议规定了其他阈值，用 `--conf-thres` 或 `--fixed-conf-thres` 覆盖。
- `--conf-floor 0.001` 只用于计算 AP 时保留低置信度候选、构造完整 PR 曲线；它不是最终汇报的固定推理阈值。
- `--nms-iou-thres 0.6` 与 YOLOv5 测试时的默认 NMS IoU 一致。
- 匹配 IoU 固定为 `0.5`，对应 `mAP50`；脚本会拒绝其他 `--match-iou-thres`，避免输出名义上是 mAP50、实际上使用其他 IoU 的结果。

`summary.csv` 中的 `precision`、`recall`、`f1` 是统一固定 `conf-thres` 下的宏平均结果；`map50` 是 YOLOv5 AP@IoU=0.50 的类别平均。为便于和 YOLOv5 `val.py` 的官方日志对照，脚本还输出 `precision_best_f1`、`recall_best_f1` 和 `best_f1_confidence`。这三个字段会对每个 split（train 或 test）独立扫描自己的 PR 曲线并选择最佳 F1 点，因此 train/test 不共用这个最佳置信度；“口径可比”指计算公式一致，而不是阈值一致。

## 输出文件

- `comparison.md`：最终主结果，只有两张 Train/Test 并排表：统一固定阈值表和各自最佳 F1 点表。
- `comparison_fixed_threshold.csv`：统一固定阈值的 Train/Test 对比表。
- `comparison_best_f1.csv`：各 split 独立最佳 F1 点的 Train/Test 对比表。
- `comparison.csv`：包含固定阈值和最佳 F1 字段的宽格式兼容文件。
- `summary.csv`：每行一个 task-split，共 18 行（9 个 task × train/test）。
- `per_image_metrics.csv`：每张图片的 GT 数、预测数、TP、FP、FN、Precision、Recall、F1 及类别计数。
- `predictions.csv`：每个候选预测的类别、置信度、框、匹配 GT、IoU 以及 TP/FP 标记。
- `ground_truths.csv`：每个 GT 的类别、框、匹配预测、IoU 以及是否被检测到。
- `per_class_metrics.csv`：每个 task-split-class 的 Precision、Recall、AP50 等。
- `missing_ids.csv`：本地数据中无法解析的稳定 ID。默认严格模式会直接报错；使用 `--skip-missing-ids` 才会跳过并记录，便于复现实验时发现数据快照差异。

训练集定义为每个 task 的 `data/selected_ids.txt`，测试集定义为该 task 的 `data/test_*_ids.txt`。推理权重默认使用各 task 的 `checkpoints/best.pt`。
