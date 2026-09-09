# stroller difficult/easy nested effect

## 研究目的

本 Study 比较两个训练集对同一困难数据测试划分的实际影响：m1 在 `strollerdifficult` 的 `train1` 上训练，m2 在其中属于 `strollereasy` 的 `train2` 上训练。两个模型都在 `test1`、`test2` 和 `testhard` 上评估。

## 数据构造

- `strollerdifficult` 是 `strollerraw` 先独立做 `tau=0.99、temporal-window=10` 去冗余后的代表集。
- `strollereasy` 是筛选困难样本后的源数据再独立做相同去冗余后的代表集。
- 为避免两个独立去冗余结果的物理路径不同，easy 成员通过 `images/` 下的相对路径与 difficult 对齐；`train2/val2/test2` 是 `train1/val1/test1` 在该 easy 成员集合上的投影。
- `testhard = test1 - test2`，因此 testhard 与 test2 不重叠；`test1` 是 difficult 的完整测试集。
- 每个 split seed `41、42` 都独立构造一套 `train1/val1/test1`，比例为 `8:1:1`。

## 训练和评估协议

m1/m2 均为原始 YOLOv5 v7.0、YOLOv5s、100 epochs、640 输入、batch 64、Full Cold，从同一 `yolov5s.pt` 初始化，训练 seed 固定为 42。checkpoint 只按各自的 val1/val2 选择；比较脚本用 best checkpoint 在三个测试组上执行相同的 YOLOv5 评估。

## 比较口径

- `m2_test1` 与 `m1_test1`：easy-only 训练是否能泛化到 difficult 的完整测试分布；
- `m2_test2` 与 `m1_test2`：加入困难样本后的 m1 是否损害 easy 情形判断；
- `m2_testhard` 与 `m1_testhard`：困难子集上的直接差异。

本 Study 明确不做 m3。因此 m1 与 m2 不仅样本难度不同，训练样本数量和数据分布也不同，结果是“实际训练方案差异”的证据，不是控制样本数后的纯困难标记因果效应。结论必须同时报告各 train/val/test 样本数、交集和 `testhard` 大小。

## 输出

- `experiment/variants/`：两套 split 的 m1/m2 训练布局、评估布局、manifest 和嵌套关系审计；
- `experiment/task/`：m1/m2 的原始 Task 产物；
- `result/evaluation/`：跨 test1/test2/testhard 的统一评估和差值；
- `result/cross_evaluation/`：交叉评估的指标柱状图、m2−m1 AP 差值图和摘要；
- `logs/`：准备、校验、训练和评估日志。

交叉评估图只读取已完成的 `result/evaluation/comparison_by_split.csv`，不修改原始 Task 或评估结果。
