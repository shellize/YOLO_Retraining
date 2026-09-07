# 对照实验：时序去冗余与数据划分诊断

## 目的

这个 study 想回答两个问题：相邻帧高度重复时，删掉冗余帧能节省多少训练数据、会损失多少检测性能；以及随机按帧重新划分后出现的高分，是否来自相邻画面跨训练集和测试集造成的数据泄漏。

三个 Sequence 都采用四阶段 Full Cold。每个阶段都从 `yolov5s.pt` 重新训练，并使用截至当前阶段的累计数据，因此主要比较的是“数据布局/筛选方式”，不是热启动或回放策略。

## 子实验的主要差别

| 子实验 | 核心设置 | 最终训练图像 | 在对照中的角色 |
|---|---|---:|---|
| `full_all` | 使用原始普通划分，不做去冗余 | 7,573 | 正常全量基线 |
| `dedup_tau_0p900` | 在原划分的训练数据内，以相邻帧窗口 1、τ=0.90 聚类，每簇保留一个代表 | 3,917 | 去冗余方法 |
| `random_frame_s42` | 把帧随机重新分配到训练/验证/测试布局 | 7,573 | 数据泄漏诊断，不是公平的同划分对照 |

`random_frame_s42` 可能让相邻、近重复画面分别落入训练集和测试集，因此它的显著高分只能说明随机逐帧划分会高估泛化能力，不能作为去冗余方法优于或劣于全量训练的证据。公平的核心比较是 `full_all` 与 `dedup_tau_0p900`。

## 结果位置

- 各 Sequence 的阶段指标：`experiment/sequence/<实验名>/summary/metrics_by_task.csv`
- 各阶段原始指标：`experiment/sequence/<实验名>/tasks/<stage>/task_result.json`
- study 汇总图表与阈值记录：`result/fullcold_comparison/`
- `random_frame_s42` 各类别 AP@0.5:0.95 learning curve：`result/random_frame_per_class_learning_curve/`
