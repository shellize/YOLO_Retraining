# 对照实验：Balance 学习曲线上的两种去冗余策略

## 目的

这个 study 在既有 Balance ordered 两批次学习曲线协议上，比较两种训练集压缩策略：普通的“一簇一个代表”，以及利用标注状态保护正样本、同时大量删除纯背景的策略。目标是判断在固定验证/测试集和固定 batch 顺序下，压缩比例与标签保护方式如何影响学习曲线。

两条新 Sequence 都沿用 ordered 基线的八阶段顺序、固定 Balance holdout 和 Full Cold 训练。基线本身位于 `../0826_balance_learning_curve_2batch/`，不在本 study 内重复运行。

## 子实验的主要差别

| 实验臂 | 核心筛选规则 | 最终训练图像 | 在对照中的角色 |
|---|---|---:|---|
| ordered 原始基线 | 不去冗余 | 7,573 | 外部参照 |
| `Balance_LC2B_ordered_dedup_tau099` | τ=0.99，每簇保留一个代表 | 5,404 | 温和压缩 |
| `Balance_LC2B_ordered_tau096_mixed_release_drop_bg` | τ=0.96；删除纯背景簇，纯标注簇留一个代表，混合簇保留全部成员 | 2,378 | 激进压缩并保护混合簇 |

第二个变体同时改变了阈值、样本量和标签处理规则，因此它与 τ=0.99 的差异不能单独归因于“释放混合簇”这一项。这个 study 回答的是整套数据策展策略的工程效果，而不是单因素消融。

## 结果位置

- 子实验结果：`experiment/sequence/<实验名>/summary/metrics_by_task.csv`
- 筛选数量和簇统计：`experiment/variants/manifest_summary.json`
- 构造规则和来源说明：`README.md`
