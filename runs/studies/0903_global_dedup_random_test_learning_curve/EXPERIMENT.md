# 对照实验：全局去冗余后重新划分的学习曲线

## 目的

这个 study 想观察：如果先在完整数据池上统一去除相邻重复帧，再从保留下来的代表图中重新构造固定验证集、测试集和训练 stage，模型性能会怎样随累计训练样本数增长。

它目前只有一条 Sequence，内部对照变量是训练数据量，而不是不同算法：八个点分别使用 615、1,230、1,844、2,458、3,072、3,686、4,300 和 4,914 张累计训练图像。

## 数据流程与阶段差别

```text
全部 9,573 张图
  → 相邻帧去冗余（τ=0.99，window=1，每簇一个代表）
  → 6,914 张代表图
  → 固定随机划分 1,000 validation + 1,000 test
  → 剩余 4,914 张随机均分为 8 个训练 stage
  → 每个 stage 从 yolov5s.pt 重新 Full Cold 训练
```

| 子实验/阶段 | 主要差别 |
|---|---|
| `GlobalDedupTau099_PostSplit` 的 stage0-stage7 | 去冗余规则、验证集和测试集都不变，只逐步增加累计训练图像数 |

因为这里采用“全局去冗余后再划分”，它与普通 `self_improving.yaml` 或 Balance 划分不是同一数据协议。曲线可以说明本 study 内部的数据规模效应，但不能把数值差直接解释成去冗余方法相对既有 Balance 基线的提升。

## 结果位置

- 汇总图、CSV 和解读：`result/learning_curve/`
- Sequence 阶段指标：`experiment/sequence/GlobalDedupTau099_PostSplit__seq-full-cold__stage0-stage7__yolov5s__s42/summary/metrics_by_task.csv`
- 数据构造与样本统计：`experiment/variants/dedup_tau_0p990/manifest_summary.json`
