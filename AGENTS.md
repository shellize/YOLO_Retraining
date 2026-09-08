# YOLO Retraining 项目指南

本文件只约束这个 Git 仓库。无论 checkout 根目录叫什么，开始工作前都先用 `git rev-parse --show-toplevel` 定位仓库根，再用 `git status --short` 查看用户已有修改；不要把父工作区或另一台机器的状态当成本仓库现状，活动代码不得依赖仓库根的 basename。

## 基本原则

1. 先明确研究问题、比较变量和证据口径，再改配置或启动训练。
2. 优先复用已有框架、公共配置和公共工具；实验专用文件留在对应 Study 内。
3. 区分源码、实验输入、原始运行产物和派生分析，不用图表反推或覆盖原始结果。
4. 普通 Task/Sequence 不支持覆盖或断点续跑。已有 `task_result.json`、`sequence_result.json`、checkpoint 和日志默认只读。
5. 普通划分、`Balance_Test` 和各 Study 自建 manifest 属于不同数据协议；比较时必须明确边界。
6. YOLOv5 vendor 保持固定且干净，通过 backend bridge/runtime hook 扩展，不直接修改第三方源码。
7. 路径必须可随 Git checkout 迁移：shell 从脚本位置推导仓库根，Python 从 `__file__` 推导；活动 YAML 优先使用相对路径。不要硬编码本机 `E:/.../project_v2`、服务器 `/home/...` 或根目录名 `project_v2`。

## 长期项目地图

```text
<repo-root>/
├── src/yolo_retraining/   可复用训练框架
├── configs/               跨实验共享的默认项、模型、超参、数据和基线配置
├── scripts/               跨实验复用的启动、预检、汇总和数据维护工具
├── data/                  本地原始数据和 self_improving 数据集
├── data_analyse/          训练前的数据统计、manifest 和冗余分析方法
├── result_analysis/       可复用的训练后分析方法
├── charts/                可复用的比较/绘图脚本
├── runs/
│   ├── tasks/             非 Study 的单 Task 历史运行
│   ├── sequences/         非 Study 的 Sequence 历史运行
│   └── studies/           自包含的对照实验
├── tests/                 单元测试和可选真实后端 smoke test
├── requirements/          训练、开发、分析和 Ultralytics 环境定义
├── reports/               既有报告资产（新 Study 结果不放这里）
└── .third_party/yolov5/   固定 YOLOv5 v7.0 运行时依赖
```

Python 版本、依赖、包入口和可选后端以 `pyproject.toml` 与 `requirements/` 为准，不在本文件复制容易过期的版本号。默认研究边界是原始 YOLOv5；Ultralytics 是可选后端。

## Study 是对照实验的完整边界

`runs/studies/` 下的每个一级目录代表一个完整 Study，而不是一个 Sequence 或 Task。Study 名称必须以四位月日开头：

```text
MMDD_<descriptive_study_name>/
```

例如 9 月 4 日创建的实验使用 `0904_...`。日期表示该 Study 建立/启动的日期；不要用整理文件的日期冒充实验日期。

每个 Study 使用以下结构，暂时没有内容的目录也可以为空：

```text
runs/studies/MMDD_<study>/
├── EXPERIMENT.md
├── config/
├── logs/
├── result/
│   └── <analysis_name>/
├── experiment/
│   ├── sequence/
│   ├── task/
│   └── variants/
└── scripts/
```

各目录职责：

- `EXPERIMENT.md`：先写研究目的，再写共同协议、各实验臂/Sequence/Task 的主要差别，以及结论不能越过的边界。不要逐项抄录所有参数。
- `config/`：只放该 Study 专用的数据布局和 Task/Sequence 配置；真正跨 Study 复用的基线仍放公共 `configs/`。
- `logs/`：GPU 等待、构造、训练和汇总日志。
- `result/<analysis_name>/`：派生图、CSV、JSON、Markdown 等分析结果。同一 Study 做另一种分析时新建另一个子目录，不把不同分析混在一起。
- `experiment/sequence/`：该 Study 启动的 Sequence 原始输出。
- `experiment/task/`：该 Study 直接启动的 Task 原始输出；Sequence 内部 Task 仍留在对应 Sequence 的 `tasks/` 中。
- `experiment/variants/`：Study 专用的预处理数据、manifest、layout、cluster audit 和 protocol 快照。
- `scripts/`：该 Study 专用的启动器、等待器、数据构造器、汇总器和一次性分析脚本。

Study 专用配置、脚本和结果不得再分散回公共 `configs/`、`scripts/`、`charts/` 或 `result_analysis/`。移动历史运行目录后，原始 YAML、JSON 和日志中可能保留服务器上的旧绝对路径；这些是运行时证据，不手改。新运行使用的活动配置和脚本则必须指向新目录。

Study 会持续增删，根规则文件不登记具体 Study 清单。判断某个 Study 做了什么，始终先读它自己的 `EXPERIMENT.md`。

多个 Study 共用的大型、确定性方法缓存可以保留在对应方法的缓存目录，由 Study 配置或脚本引用，不要求重复复制；必须在 `EXPERIMENT.md` 说明来源和复用条件。共享缓存不是一个 Study，Study 专用 manifest、layout 和 protocol 仍放 `experiment/variants/`。

## 公共目录边界

### `src/yolo_retraining/`

核心执行链：

```text
YAML / --set
  -> config.py
  -> TaskRunner 或 SequenceRunner
  -> data registry/scope
  -> selection policy
  -> YOLOv5 或 Ultralytics backend
  -> Task/Sequence 原始产物
  -> Study result 或可复用分析方法
```

主要模块：

- `run.py`：CLI 入口。
- `config.py`：include 合并、覆盖、相对路径解析和校验。
- `engine/task_runner.py`、`engine/sequence_runner.py`：单次与多阶段执行。
- `engine/output.py`：输出命名与禁止覆盖。
- `data/`：registry、manifest、scope 和 loader；不是物理数据目录。
- `policies/selection/`：full、random replay 和 selection study 策略。
- `backends/yolov5/`、`backends/ultralytics/`：检测器适配。
- `evaluation/`：指标和训练代价。
- `visualization/`：本地 TensorBoard 写入。

### 公共 `configs/`

- `defaults.yaml`：默认训练、评估 artifact、best metric 和固定置信度诊断设置。
- `data/`：跨 Study 复用的数据协议和示例布局。
- `model/`：检测器模型配置。
- `hyp/`：公共超参数组合。
- `task/`：Full Cold、Full Warm、Current Only、Random Replay 和合并 stage 的公共 Task。
- `sequence/`：标准基线、selection study 和跨 Study 共享的 Sequence 基础配置。

配置 include 和相对路径按配置文件位置解析。移动 YAML 后必须重新加载验证，不能只看文本路径似乎正确。

四条标准 Sequence 基线：

| 基线 | 训练数据 | 后续初始化 |
|---|---|---|
| `full_cold` | 截至当前阶段的全部已见数据 | 每阶段从预训练权重冷启动 |
| `full_warm` | 全部已见数据 | 上一 Task 的 `best.pt` |
| `current_only` | 当前阶段数据 | 上一 Task 的 `best.pt` |
| `random_replay` | 当前数据加固定随机历史子集 | 上一 Task 的 `best.pt` |

Task 级 warm 配置可能使用父任务 `last.pt`，标准 Sequence 使用 `best.pt`；结果说明中必须写清 checkpoint 语义。

### 公共 `scripts/`

只保留跨实验复用入口：标准基线/Balance Test 启动器、selection study 启动器和 preflight、`select_best_task.py`、`normalize_batch_names.py`、通用冗余汇总与协议校验等。

具体 Study 的运行命令从其 `scripts/` 启动。新增 Study 时，不要再把它的 `run_*.sh` 放回公共 `scripts/`。

### `data/` 与 `data_analyse/`

物理数据只放在根 `data/`：

- `data/raw/iamge/`、`data/raw/label/` 是现有真实拼写，不擅自更名。
- `data/self_improving/images/`、`labels/` 按物理 batch 保存。
- `data/self_improving/annotation_archive/` 保存原始 XML/JSON 归档。
- `data/classes.txt` 保存类别顺序。

不要展平 batch，不覆盖原始标注，不把大数据提交 Git。数据组合通过公共 layout 或 Study 的 `experiment/variants/` manifest 完成。

`data_analyse/` 只保存可复用的数据研究方法：

- 批次统计方法：统计图像、标注和类别。
- 自定义数据方法：构建/校验 manifest 和数据布局，不复制原图。
- 冗余分析方法：固定特征、时序相似度、聚类和代表样本选择。

冗余分析是训练前审计，不允许用 test mAP 反向选择 τ。相邻帧主协议使用 `temporal-window=1`；更宽窗口或全局近邻应作为不同诊断标明。

### `charts/`、`result_analysis/` 与 `reports/`

- `charts/` 只放可跨 Study 复用的图表/比较脚本。
- `result_analysis/` 只放可复用的模型结果分析方法。
- Study 专用输出必须写入 `runs/studies/<study>/result/<analysis_name>/`；Study 专用分析入口放该 Study 的 `scripts/`。
- 这两个公共目录里已经存在的旧图、CSV 和时间戳结果本轮不整理；它们不是新的目录范式。
- `reports/` 中的既有报告资产不作为新 Study 的结果目录。

## 原始运行产物与取数规则

单 Task 的 canonical 证据包括：

```text
task.yaml
task_status.json
task_result.json
data/*_ids.txt
selection/
checkpoints/{last.pt,best.pt}
metrics/{train_history.csv,evaluation.json,cost.json}
backend/<backend>/
tensorboard/
logs/
```

Sequence 还包括 `sequence.yaml`、`sequence_status.json`、`sequence_result.json`、内部 `tasks/` 和 `summary/*.csv`。

取数顺序：

1. 先检查 status 是否完成以及完成 Task 数。
2. 最终指标和权重路径读 `task_result.json`。
3. 阶段趋势读 Sequence 的 `summary/metrics_by_task.csv`，必要时回到各 Task 核对。
4. 实际训练样本读 `data/selected_ids.txt`，筛选规则读 `selection/summary.json`。
5. 训练曲线读 `metrics/train_history.csv` 或 TensorBoard event。
6. 固定置信度诊断读 backend evaluation 下的 `confidence_sweep.json`；它不等于各自最优 F1。
7. 图表和 Markdown 是派生交付物，不能作为唯一事实来源。

不得手改原始结果，也不得复制已有 runs 冒充新实验。目录搬迁不会自动重写历史文件中的绝对路径，分析程序应优先根据当前 Study 根定位文件，并把历史路径只当 provenance 文本。

## Git 与产物同步

- Git 只同步源码、配置、启动/分析脚本、Study 协议和 `experiment/variants/`；物理数据、原始训练输出、日志、权重、派生分析和报告不进入 Git。
- 服务器是训练、推理和重计算型分析产物的写入端；本地保留完整副本供浏览和轻量分析。需要更新本地副本时，手动运行 `scripts/sync_artifacts_from_beidou.sh`，从服务器单向镜像到本地，不反向上传；默认不启用后台定时同步。
- 本地 `.artifact_sync/config.env` 保存机器专用的服务器地址和 SSH 命令，不提交 Git。同步脚本不使用 `--delete`，因此服务器删除或整理文件不会自动删除本地副本。
- 手动镜像只覆盖 `runs/tasks/`、`runs/sequences/`、Study 的 `experiment/{sequence,task}/`、`logs/`、`result/`、分析方法的 `results/`、生成的 chart payload 和 `reports/`。Study 的配置、脚本和 variants 始终只由 Git 更新。
- Git 拉取不等于产物已同步；报告当前状态时分别核对 Git HEAD 和最近一次产物镜像日志。

## 环境、运行与验证

训练环境默认 `yolo-retraining-v5`，结果分析环境默认 `yolo-result-analysis`。不要为了画图或 ONNX 检查修改服务器训练环境。

常用入口：

```bash
conda run --no-capture-output -n yolo-retraining-v5 \
  python -m yolo_retraining.doctor --project-root "$PWD"

yolo-retraining task --config configs/task/full_cold.yaml
yolo-retraining sequence --config configs/sequence/full_warm.yaml

conda run --no-capture-output -n yolo-retraining-v5 python -m pytest -q
```

使用 `--set KEY=VALUE` 做少量单次覆盖，不为一两个参数复制整份 YAML。长任务启动前检查输出冲突、数据协议、seed、checkpoint、GPU 和磁盘；需要等待 GPU 的 Study 应使用自己的 waiter，并检查 tmux、进程、GPU、日志和结果文件，不能仅凭显存瞬时值判断状态。

测试按改动范围选择：配置、数据/manifest、Task/Sequence、selection、backend、分析方法和 Study 路径各自运行对应测试；跨目录迁移还要运行完整无 GPU 测试集。

真实 YOLOv5、Ultralytics、ONNX 或多 GPU smoke test 与默认无 GPU 单测分开。只有实际运行过的测试才能写成已验证，不在本文件固定保存容易过期的 pass 数量。

## 长期操作边界

1. `.third_party/`、`data/`、Study 原始结果、checkpoint、方法缓存和派生图表属于依赖、数据或生成资产；不要为了清理目录擅自删除。
2. `.gitignore` 会忽略数据、第三方依赖、权重和一部分生成资产；Git 状态不能证明这些文件不存在。
3. 数据重命名、训练、重新评估、导出和 manifest 重建都会写磁盘；纯审阅或解释请求不自动执行。
4. 新增或移动 Study 后，至少验证：七个标准目录存在、活动 YAML 可加载、脚本语法通过、相关测试通过、原始 Task/Sequence 数量未减少。
5. 临时故障、当前实验列表、一次测试的 pass 数和机器专用路径不写入本文件；它们应记录在对应 README、`EXPERIMENT.md`、issue 或运行日志中。
