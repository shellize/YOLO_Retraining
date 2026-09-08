# difficult stroller window=10 learning curve

## 研究目的

本 Study 观察在未做困难样本筛选、但已经统一去冗余的 `strollerdifficult` 上训练时，随着训练数据逐步增加，整体 stroller AP 如何变化。它是后续 m1/m2 困难样本影响实验的全量数据基线。

## 数据协议

- `strollerraw`：服务器 `data/stroller_protocol_0908/stroller_raw_source` 中的全部 stroller 图片。
- `strollerdifficult`：对 `strollerraw` 独立执行 `tau=0.99、temporal-window=10` 后保留的代表样本。
- 去冗余输出来自 `data_analyse/stroller_protocol_0908/results/difficult/`；本 Study 不复用旧 Study 的 manifest、划分或权重。
- 每个划分按 `train:test:val=8:1:1`，训练集再按 source manifest 顺序分成 8 个 stage。
- 划分模式为两个随机 split seed `41、42`，以及一个 source-order 对照。顺序对照只用于检查 manifest 顺序敏感性，不作为独立随机重复。

## 训练协议

所有 Sequence 都使用公共 Full Cold learning-curve 设置：原始 YOLOv5 v7.0、YOLOv5s、每个 stage 100 epochs、输入 640、batch 64、GPU 0；每个 stage 从 `yolov5s.pt` 冷启动，并使用训练 seed `42+stage_index`。验证集只用于选择 checkpoint，最终曲线读取各 Task 的 `best.test` 指标。

## 解释边界

该 Study 的测试集来自各自 split，不能用来证明困难标记的因果损害；它只提供困难样本未筛选时的完整数据性能基线。困难样本的加入效应由 `0908_stroller_difficult_easy_nested_effect` 中 m1/m2 的交叉测试进一步观察。m3 不在本实验设计中。

## 输出

- `experiment/variants/`：去冗余 manifest 的 train/val/test/stage 划分；
- `experiment/sequence/`：原始 Sequence/Task 产物；
- `result/learning_curve/`：从 `task_result.json` 汇总的曲线和 CSV；
- `logs/`：准备、校验和训练日志。

本次只写入准备与启动脚本，不启动训练。
