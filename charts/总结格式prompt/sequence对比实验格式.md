# Sequence 对比实验总结 Prompt

下面的 Prompt 用于比较 `runs/sequences` 中多种序列训练方法，并生成阶段折线图、完整指标表和总结报告。使用时替换方括号中的内容；没有明确给出的部分必须先根据实际结果文件核验，不能猜测。

## 可直接使用的 Prompt

```text
请分析项目中 `runs/sequences` 下的 Sequence 实验结果，并把本次分析使用的脚本和所有产物统一放在：

`charts/[本次分析工作目录名]/`

不要修改 `runs` 中的任何原始训练结果，也不要把脚本或临时文件散落到其他目录。

一、首先核验实验状态和比较范围

1. 扫描 `[需要比较的实验目录或目录匹配规则]`。
2. 检查每条 Sequence 的：
   - `sequence_status.json`
   - `sequence_result.json`
   - 各 Stage 的 `task_status.json`
   - 各 Stage 的 `task_result.json`
3. 只有同时满足以下条件的 Sequence 才能作为完整折线参与比较：
   - Sequence 状态为 `completed`；
   - Stage 0～Stage 3 均能找到可用结果；
   - 每个 Stage 都存在目标 checkpoint 和 test split 的指标。
4. 对 `running`、`failed`、缺少 Stage 或缺少 `task_result.json` 的实验：
   - 不得用训练中间指标冒充最终测试指标；
   - 不得静默忽略；
   - 必须在报告中列出实验名、实际状态和未纳入原因。
5. 根据实际完成结果，明确写出“方法”和“baseline”的准确映射。不要只根据我的自然语言描述猜测方法数量。
6. 检查所有实验是否使用相同的数据划分、模型、imgsz、seed、batch、测试集和 checkpoint 选择标准。如果存在差异，先列出差异，并禁止直接归因于样本选择方法。

二、统一指标口径

1. 默认读取每个 Stage 的：

   `task_result.json -> metrics.best.test`

2. 使用以下三个指标：
   - `map50`
   - `recall`
   - `precision`
3. 不要混用 `last/test`、`best/val`、训练历史中的指标或独立优化阈值后的指标。
4. 如果实验复用了已有 Stage 0 checkpoint，应从对应的 bootstrap Stage 0 `task_result.json` 中读取 Stage 0 指标，并在报告中说明哪些方法共享 Stage 0。
5. 表格保留至少四位小数；CSV 中保留至少八位小数。

三、生成三张折线图

分别生成：

1. `map50.png`
2. `recall.png`
3. `precision.png`

绘图要求：

- 横轴为 `Stage 0`、`Stage 1`、`Stage 2`、`Stage 3`；
- 纵轴为对应的 `best/test` 指标，范围按全部真实观测值自动留出合理边距；
- 每条折线代表一种方法；
- baseline 使用更粗或更醒目的线型，但不能遮挡其他方法；
- 不同方法同时使用颜色、线型和 marker 区分，不能只靠颜色；
- 三张图的方法颜色映射必须保持一致；
- 图例包含完整方法名；
- 不在折线附近堆叠大量数值标签，具体数值放在表格中；
- 图片应清晰、无裁切、图例不覆盖曲线。

四、生成具体指标表

生成 `metrics_tables.md`，分别提供三张表：

1. mAP50 表；
2. Recall 表；
3. Precision 表。

每张表格式：

| Method | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Mean |
|---|---:|---:|---:|---:|---:|

同时生成长表格式的 `metrics.csv`，至少包含：

- `method_key`
- `method`
- `family`
- `is_baseline`
- `stage`
- `stage_label`
- `map50`
- `recall`
- `precision`
- `task_result`

`task_result` 必须记录实际数值来源路径，保证每个值可追溯。

五、比较和解释

生成 `comparison_summary.md`，至少包括：

1. 本次比较采用的 checkpoint、split、baseline 和方法列表；
2. 未完成或未纳入的实验及原因；
3. 每个 Stage、每项指标的最佳方法和值；
4. 每种方法在 Stage 3 相对 baseline 的：
   - ΔmAP50
   - ΔRecall
   - ΔPrecision
5. 四个 Stage 的平均 mAP50、Recall 和 Precision；
6. 重点分析 Precision 和 Recall 是否存在权衡：
   - Precision 提高但 Recall 下降时，不能称为模型整体性能提高；
   - Recall 提高但 mAP50 下降时，要说明提升是否稳定、是否只出现在个别 Stage；
7. 判断是否有方法在多个 Stage 上稳定优于 baseline；
8. 单 seed 结果只能作为描述性比较，不能声称具有统计显著性。

结论必须由表格中的数值直接支持。需要重点回答：

- 哪种方法的 mAP50 最稳定、最终最高？
- 哪种方法 Recall 最好，是否牺牲 Precision？
- 哪种方法 Precision 最好，是否只是因为预测更保守？
- 哪种方法在 Stage 3 最接近或超过 baseline？
- 样本筛选方法是否出现决定性改进，还是仍不如全量训练？

六、生成一个包含全部内容的最终 Markdown

生成 `complete_report.md`，按以下顺序组织：

1. 标题和指标口径；
2. mAP50 图片：`![mAP50 comparison](map50.png)`；
3. Recall 图片：`![Recall comparison](recall.png)`；
4. Precision 图片：`![Precision comparison](precision.png)`；
5. 比较范围和缺失实验说明；
6. 每个 Stage 的最佳方法表；
7. Stage 3 相对 baseline 差值表；
8. 主要比较结论；
9. 三张完整指标表；
10. `metrics.csv` 和分析脚本的相对链接。

最终 Markdown 中所有图片和文件链接必须使用相对路径，使整个工作目录复制到其他位置后仍可正常查看。

七、脚本和验证要求

在工作目录中保存一个可重复运行的分析脚本，例如：

`compare_methods.py`

脚本必须：

- 显式记录方法名与 Sequence 目录的映射；
- 在读取前验证 Sequence 是否完成；
- 验证每个 Stage 恰好对应一个 `task_result.json`；
- 验证 `metrics.best.test` 和三个目标指标均存在；
- 自动生成三张图片、CSV、指标表、比较摘要和最终报告；
- 重复执行时覆盖派生结果，但绝不修改 `runs`；
- 在数据缺失或状态不完整时明确报错，而不是生成不完整但看似正常的图。

生成后执行以下检查：

1. 分析脚本能够正常运行；
2. 三张图片实际存在且大小合理；
3. 人工查看三张图，确认坐标、曲线、marker、图例没有重叠或裁切；
4. `complete_report.md` 中引用的所有相对路径都存在；
5. CSV、Markdown 表格和折线图使用的是同一批数值；
6. 最终回复中给出工作目录和 `complete_report.md` 的可点击路径，并简要说明最重要的三到五条结论。
```

## 推荐的工作目录结构

```text
charts/[本次分析工作目录名]/
├── compare_methods.py
├── map50.png
├── recall.png
├── precision.png
├── metrics.csv
├── metrics_tables.md
├── comparison_summary.md
└── complete_report.md
```

## 使用时建议补充的信息

为了减少歧义，调用这个 Prompt 时最好同时说明：

- baseline 的准确 Sequence 目录名；
- 需要比较的实验目录名或前缀；
- 使用 `best` 还是 `last` checkpoint；
- 使用 `test`、`val` 还是其他固定测试组；
- 是否只比较完整实验；
- 未完成实验是排除、等待完成，还是单独显示为不完整曲线；
- 图片和报告使用中文还是英文标签。
