# 对照实验：Balance 两批次学习曲线与顺序敏感性

## 目的

这个 study 用固定的 Balance 验证集和测试集，观察训练数据每增加两个物理 batch 时，Full Cold 性能如何变化；同时通过两种随机 batch 顺序，判断学习曲线的中间形状是否过度依赖某一种数据到达顺序。

三个 Sequence 最终都会看到同一批 7,573 张训练图像，模型和训练方式也一致。它们真正的变量是“八个 stage 中，各批次以什么顺序加入”。因此重点应比较中间七个学习曲线点，而不是只看最终点。

## 子实验的主要差别

| 子实验 | Batch 到达方式 | 在对照中的角色 |
|---|---|---|
| `Balance_Learning_Curve_ordered` | 按既定批次顺序，每两个 batch 组成一个 stage | 主学习曲线 |
| `Balance_Learning_Curve_random_s41` | 使用数据顺序种子 41 重排 batch，再按两个 batch 一组累加 | 顺序敏感性对照 1 |
| `Balance_Learning_Curve_random_s43` | 使用数据顺序种子 43 重排 batch，再按两个 batch 一组累加 | 顺序敏感性对照 2 |

这里的 `s41`、`s43` 指数据顺序随机种子；模型训练仍遵循 Sequence 自身的 stage seed schedule。三条曲线的最终训练集相同，所以该 study 主要用于判断“小数据阶段的结论是否稳健”。

## 结果位置

- 各 Sequence 的阶段指标：`experiment/sequence/<实验名>/summary/metrics_by_task.csv`
- 已有跨 Sequence 绘图脚本：`../../charts/balance_learning_curve_2batch/generate_report.py`
