# YOLO Retraining 项目指南

## 项目定位

本项目是研究型 YOLO 顺序重训练框架。实验由 YAML 描述，核心执行单元分为：

- `Task`：一次独立训练，显式指定当前数据、候选数据、初始化方式和父任务。
- `SequenceExperiment`：按 `stage0 -> stageN` 展开多个完整 Task，并显式传递上一阶段结果。

默认检测器是固定提交 `915bbf294bb74c859f0b41f1c23bc395014ea679` 的原始 YOLOv5 v7.0；`ultralytics` 是可选后端，不是默认依赖。

## 开始工作前

1. 先阅读本文件和与任务最接近的目录 README。
2. 优先复用已有配置、方法或脚本，不要在确认没有现成入口前新建重复工具。
3. 区分代码/协议与生成资产：`src/`、`configs/`、`scripts/` 是实现；`data/`、`runs/` 和各类结果子目录是本机数据或产物。
4. 工作树可能已有用户修改。先运行 `git status --short`，保留无关改动。
5. 普通 Task/Sequence 不支持覆盖或断点续跑；不要删除、覆盖或擅自移动已有运行目录。
6. 面向某次具体分析的生成脚本必须与它生成的图像、CSV、JSON、Markdown 等结果放在同一个结果目录中，不得放入 `scripts/`。该规则同样适用于 `charts/`、`data_analyse/`、`result_analysis/`、`reports/` 等目录；`scripts/` 只保留实验启动器、预检、GPU 等待器和不绑定具体结果目录的通用工具。

## 项目地图

```text
configs/           实验协议、模型、数据布局、Task 和 Sequence 配置
src/               核心训练框架
scripts/           实验启动器、预检、GPU 等待器和通用工具
data_analyse/      训练前的数据诊断与自定义数据布局工具
result_analysis/   训练后的模型诊断与人工复核工具
charts/            针对既有实验的图表/报告生成器（多为硬编码、本地生成）
requirements/      训练、开发、Ultralytics、结果分析环境
tests/             单元测试和可选 GPU smoke test
data/              本地数据集，不是源码
runs/              Task、Sequence、Study 运行产物
.third_party/      固定 YOLOv5 源码、权重和下载缓存
logs/              外围日志
```

## 核心执行链

```text
配置 YAML / --set 覆盖
  -> src/yolo_retraining/config.py
  -> TaskRunner 或 SequenceRunner
  -> 数据 registry/scope
  -> selection policy
  -> YOLOv5 或 Ultralytics backend
  -> checkpoints、metrics、cost、TensorBoard、TaskResult
  -> result_analysis/ 或 charts/
```

关键实现：

- CLI：`src/yolo_retraining/run.py`
- 配置合并、相对路径解析和校验：`src/yolo_retraining/config.py`
- 单任务执行：`src/yolo_retraining/engine/task_runner.py`
- 多阶段执行：`src/yolo_retraining/engine/sequence_runner.py`
- 输出命名与禁止覆盖：`src/yolo_retraining/engine/output.py`
- 数据加载和稳定 ID：`src/yolo_retraining/data/`
- 样本选择：`src/yolo_retraining/policies/selection/`
- 后端：`src/yolo_retraining/backends/`
- 持续学习汇总指标：`src/yolo_retraining/evaluation/metrics.py`

## 按任务选择入口

### 环境检查

```bash
conda run --no-capture-output -n yolo-retraining-v5 \
  python -m yolo_retraining.doctor --project-root "$PWD"
```

- 首次构建：`requirements/bootstrap.sh` 或 `requirements/bootstrap.ps1`
- 复用已有环境并重装 editable package：`requirements/use_existing_env.sh` 或 `.ps1`
- 独立结果分析环境：`requirements/bootstrap_analysis.ps1`
- 服务器 CPU/内存/GPU 快照：根目录本地工具 `check_server_resources.sh`

### 启动训练

```bash
yolo-retraining task --config configs/task/full_cold.yaml
yolo-retraining sequence --config configs/sequence/full_warm.yaml
```

使用 `--set KEY=VALUE` 做单次覆盖，避免为少量参数复制 YAML，例如：

```bash
--set backend.params.device=0
--set backend.params.imgsz=960
--set data.layout=/absolute/path/layout.yaml
```

优先选择 `configs/sequence/` 运行完整阶段实验。`configs/task/` 适合单次训练；warm/replay Task 需要显式设置 `task.parent_result`。注意：当前 warm 类 Task YAML 使用父任务 `last.pt`，标准 Sequence 使用父任务 `best.pt`，汇报时不要混淆口径。

### 四条标准基线

| 基线 | 训练数据 | 后续初始化 |
|---|---|---|
| `full_cold` | 全部已见组 | 每阶段预训练权重冷启动 |
| `full_warm` | 全部已见组 | 上一 Task `best.pt` |
| `current_only` | 当前组 | 上一 Task `best.pt` |
| `random_replay` | 当前组 + 固定随机历史子集 | 上一 Task `best.pt` |

标准配置在 `configs/sequence/{full_cold,full_warm,current_only,random_replay}.yaml`。双 GPU 固定基线启动器在 `scripts/run_*baseline*.sh` 和 `scripts/run_full_*_*.sh`。

### 样本选择研究

配置在 `configs/sequence/selection_study/`，实现位于 `src/yolo_retraining/policies/selection/study.py`：

- `positive_only`：真实标签框数大于零。
- `pred_positive`：教师模型产生检测结果。
- `error_hard`：教师模型存在 FP 或 FN。
- `gradnorm_topk`：按检测头梯度范数取 Top-K，预算来自 `error_hard` 数量。
- `random_topk`：与 `error_hard` 等预算的随机对照。

每种策略分为累计冷启动 `cum_cold` 和当前数据热启动 `current_warm`。批量启动使用：

- `scripts/preflight_selection_study.py`：检查 bootstrap、teacher checkpoint 和输出冲突。
- `scripts/run_selection_study.sh`：串行运行全部配置，完成项自动跳过。
- `scripts/run_selection_study_when_gpu0_idle.sh`：GPU0 连续空闲后启动，并用 `flock` 防重复。

### 创建或检查数据布局

首选 `data_analyse/custom_dataset/custom_dataset.py`，它不复制或移动原图：

```bash
python data_analyse/custom_dataset/custom_dataset.py validate --layout <layout.yaml>
python data_analyse/custom_dataset/custom_dataset.py write-manifest ...
python data_analyse/custom_dataset/custom_dataset.py build-layout ...
python data_analyse/custom_dataset/custom_dataset.py random-reassign ...
python data_analyse/custom_dataset/custom_dataset.py random-split ...
```

标准数据布局：

- `configs/data/self_improving.yaml`：原始批次协议。
- `configs/data/self_improving_balance_test.yaml`：平衡 holdout 协议。

布局只映射物理目录，不移动图像。稳定样本 ID 是 `group_id::relative/image/path.jpg`。新布局训练前必须校验路径、标签、类别和跨组重叠。

### 连续帧冗余分析

- 方法：`data_analyse/dataset_redundancy/redundancy_analysis.py`
- 完整编排：`scripts/run_fullcold_redundancy_shift_study.bash`
- shell 兼容入口：`scripts/run_fullcold_redundancy_shift_study.sh`
- 最终汇总：`scripts/summarize_fullcold_data_study.py`
- 旧三阈值协议校验器：`scripts/validate_redundancy_protocol.py`

当前完整编排器扫描 `0.900-0.970` 候选阈值，选择保留率最接近 `3/7` 的一个阈值再训练；`data_analyse/dataset_redundancy/README.md` 中固定三个阈值的描述已经落后，执行时以脚本为准。

### 训练结果分析

- 训练/测试过拟合：`result_analysis/train_test_overfitting/analyze_train_test_overfitting.py`
  - 比较固定置信度和各 split 最佳 F1。
  - 输出逐图 TP/FP/FN、逐预测、逐 GT、逐类 AP 和缺失 ID。
- FP/FN 人工复核：`result_analysis/fp_fn_review/build_fp_fn_review.py`
  - 从完成 Task 构造 AnyLabeling 目录。
  - 生成 GT/TP/FP 对比 JSON、manifest、ONNX 配置和错误样本链接。
  - ONNX/ONNX Runtime 放在独立 `yolo-result-analysis` 环境，避免污染训练环境。

### 图表与报告

- `charts/4balance_baseline/generate_report.py`：四基线指标、折线图和 Markdown 报告。
- `charts/selection_study_comparison/compare_methods.py`：Selection Study 对比。
- `charts/selection_study_comparison/generate_fixed_confidence_cache.py`：缺失固定置信度产物时重新评估。
- `charts/总结格式prompt/sequence对比实验格式.md`：现有总结格式参考。

`charts/` 被 Git 忽略，且当前脚本中的 run 名称和方法列表多为硬编码。只有目标实验与既有目录一致时才能直接运行；泛化到新实验前先参数化输入，避免复制粘贴整份脚本。

### 其他小工具

- `scripts/select_best_task.py`：按 `metrics/train_history.csv` 中指定验证指标选择完成 Task。
- `scripts/normalize_batch_names.py`：将 `0720_01..09` 改为 `0720_1..9` 并重写数据 YAML。它会修改数据，只能在明确授权并核对目标根目录后运行。

## 运行产物地图

单个 Task：

```text
<task>/
├── task.yaml
├── task_status.json
├── task_result.json
├── data/*_ids.txt
├── selection/
├── checkpoints/{last.pt,best.pt}
├── metrics/{train_history.csv,evaluation.json,cost.json}
├── backend/<backend>/
├── tensorboard/
└── logs/error.txt
```

Sequence：

```text
<sequence>/
├── sequence.yaml
├── sequence_status.json
├── sequence_result.json
├── tasks/000__stage0__...
└── summary/{metrics_by_task.csv,cost_by_task.csv,selection_by_task.csv}
```

查找顺序：

- 最终指标或权重路径：`task_result.json`
- 每阶段整体趋势：Sequence 的 `summary/`
- 实际训练样本：`data/selected_ids.txt`
- 样本筛选细节：`selection/summary.json`、`sample_scores.jsonl`
- 训练曲线：`metrics/train_history.csv` 或 `tensorboard/`
- 固定置信度曲线：后端 evaluation 目录的 `confidence_sweep.json`
- 失败原因：`task_status.json` / `sequence_status.json` 和 `logs/error.txt`

## 验证要求

无 GPU 默认测试：

```bash
conda run --no-capture-output -n yolo-retraining-v5 python -m pytest -q
```

当前基线为 `63 passed, 5 skipped`。被跳过的通常是 ONNX、真实 YOLOv5、Ultralytics 和双 GPU smoke test。只有相关改动才启用对应环境变量运行昂贵测试。

修改范围对应测试：

- 配置：`tests/test_config.py`
- 数据布局/manifest：`tests/test_data_*.py`、`tests/test_custom_dataset.py`
- 选择策略：`tests/test_selection_policies.py`
- Task/Sequence：`tests/test_task_runner.py`、`tests/test_sequence_runner.py`
- 冗余方法：`tests/test_dataset_redundancy.py`
- 结果分析：`tests/test_fp_fn_review.py`
- YOLOv5 后端：`tests/test_yolov5_backend.py`、`tests/test_yolov5_artifacts.py`

## 已知问题与操作边界

1. `scripts/run_full_cold_stage0.sh` 当前在定义 `PROJECT_ROOT` 前引用它，在 `set -u` 下会立即失败；修复前不要直接使用。
2. `data_analyse/class_distribution/` 当前不可用：缺少包级 `__main__.py`，默认数据根路径错误，且预期 XML 目录与当前 `data/self_improving/archive/0720_N/xml` 不一致。
3. `result_analysis` 下带时间戳目录、`runs/`、`data/`、冗余结果和图表大多是生成资产；不要把它们当作实现入口，也不要为“清理”擅自删除。
4. `normalize_batch_names.py`、新训练、重新推理和 ONNX 导出都会写入磁盘；纯分析请求不要自动执行这些动作。
5. 普通 Task/Sequence 的输出目录若已存在会拒绝运行。Study 脚本可能实现自己的完成检测和缓存复用，应先阅读相应脚本的语义。
6. YOLOv5 源码必须保持固定提交且干净；框架不会自动 reset 或覆盖不匹配的 `.third_party/yolov5`。
7. 默认优先使用 YOLOv5 后端。只有用户明确要求 YOLOv8/YOLO11 或现代 Ultralytics 行为时，才安装 `requirements/ultralytics.lock` 并选择对应模型配置。
