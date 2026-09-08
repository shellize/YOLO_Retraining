# 对照实验：全局去冗余后学习曲线与顺序切分

## 目的

本 Study 观察：先在完整数据池上统一去除相邻重复帧，再按不同数据切分协议构造固定验证集、测试集和训练 stage 时，mAP 随累计训练图像数的曲线如何变化。已有主线和随机切分对照保留不变；新增一个尽量保持去冗余输入顺序的顺序切分对照。

## 共同协议

    全部 9,573 张图
      → 相邻帧去冗余（τ=0.99，window=1，每簇一个代表）
      → 6,914 张代表图
      → 按指定切分协议分配 train/val/test
      → 4,914 张训练图分为 8 个连续 stage
      → 每个 stage 从 yolov5s.pt 重新 Full Cold 训练 100 epochs

所有实验臂均使用 YOLOv5s、640 输入尺寸、batch 64、相同的训练 seed schedule（42–49）和相同的评价口径。训练阶段仍使用 YOLOv5 默认的每 epoch DataLoader shuffle；本次新增对照只改变切分前后的样本分配顺序，不把优化器的 batch 顺序作为额外变量。

| 实验臂 | 切分协议 | Sequence 训练 seed | 作用 |
|---|---|---:|---|
| `GlobalDedupTau099_PostSplit` | split seed 42 随机切分 | 42–49 | 当前主线 |
| `GlobalDedupTau099_PostSplit_RandomS41` | split seed 41 随机切分 | 42–49 | 随机切分对照 |
| `GlobalDedupTau099_PostSplit_RandomS43` | split seed 43 随机切分 | 42–49 | 随机切分对照 |
| `GlobalDedupTau099_Ordered` | 保持去冗余输入的 canonical path 顺序，按连续区间切分 | 42–49 | 顺序切分对照 |

顺序对照沿用本 Study 原有的区间顺序：先取 train 的 80%，再取 val 的 10%，最后取 test 的 10%；训练区间内部按原顺序分成 8 个 stage。这里的“原顺序”是冗余分析输入记录的 canonical path 顺序，而不是依赖文件系统返回顺序。顺序切分会让相邻场景更可能落在同一数据区间，因此它不是随机切分的替代，而是用于检查场景顺序对 learning curve 的影响。

## 结果位置

- 主线和随机切分结果：`experiment/sequence/`
- 顺序布局：`experiment/variants/dedup_tau_0p990_ordered/`
- 顺序配置：`config/global_dedup_tau099_ordered.yaml`
- 随机性比较：`result/randomization_comparison/`
- 顺序对照综合曲线：`result/randomization_comparison/learning_curve_with_ordered_control.png`
- 对照启动与汇总：`scripts/run_randomization_controls.sh`、`scripts/run_randomization_controls_server.sh`

## 启动边界

本次只加入顺序布局、配置、启动和结果汇总代码，未在本地启动新的训练。服务器启动脚本会在已有主线/随机切分结果之后运行顺序对照，并在 8 个 Task 全部完成后生成综合比较图。已有 Task/Sequence 原始结果保持只读，不覆盖不完整输出。

随机切分曲线只支持描述所列 split seed 下的差异；顺序对照也只能说明该特定 canonical path 顺序下的结果，不能外推为所有时间顺序或所有数据划分。
