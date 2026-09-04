# 对照实验：τ=0.90 去冗余中的标签处理策略

## 目的

这个 study 进一步追问：普通时序去冗余造成的性能下降，究竟只是因为样本变少，还是因为“一簇一个代表”误删了有标注图或保留了过多纯背景。为此，在同一个 Balance 固定验证/测试划分和同一个 τ=0.90 聚类基础上，对比四种标签处理策略。

四个新实验都是把 stage0-stage3 合并成一个 Full Cold Task；它们不是连续学习 Sequence。全量原始数据的 300-epoch Task 作为外部基线引用，没有在本 study 中重复训练。

## 子实验的主要差别

| 实验臂 | 簇内保留规则 | 训练图像 | 想隔离的问题 |
|---|---|---:|---|
| 全量原始基线 | 不去冗余 | 7,573 | 性能上限参照 |
| `dedup_tau_0p900` | 每簇保留默认代表 | 4,058 | 普通去冗余基线 |
| `dedup_tau_0p900_positive_representative` | 每簇仍留一个，但优先选择有标注代表 | 4,058 | 只改变代表选择，不改变样本量 |
| `dedup_tau_0p900_release_positive_clusters` | 释放含标注簇成员，纯背景簇保留代表 | 5,866 | 检查保护正样本邻域的作用 |
| `dedup_tau_0p900_release_positive_clusters_drop_pure_background` | 释放含标注簇，同时删除纯背景簇 | 3,752 | 检查正样本保护加背景压缩的整体效果 |

`experiment/task/` 中该最后一个变体存在一个短名副本和一个规范长名目录；两者的 `task_result.json` 指向同一结果、指标相同，比较时只能计一次。`result/strategy_comparison/comparison.csv` 中的 100-epoch Sequence final stage 只是辅助参照，不属于上述五个核心实验臂。

## 结果位置

- 跨实验汇总：`result/strategy_comparison/comparison.csv`、`comparison.md`
- 各变体构造与样本量：`experiment/variants/manifest_summary.json`
- 各 Task 原始结果：`experiment/task/<实验名>/task_result.json`
