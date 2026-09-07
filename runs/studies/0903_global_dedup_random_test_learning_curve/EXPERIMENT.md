# 对照实验：全局去冗余后随机划分的学习曲线

## 目的

这个 Study 观察：先在完整数据池上统一去除相邻重复帧，再随机构造固定验证集、测试集和训练 stage 时，mAP 随累计训练图像数的曲线是否依赖于划分所用的随机数。

主线使用 split seed 42；新增两个对照使用 split seed 41 和 43。三条线都使用同一套去冗余规则、同一模型和 Full Cold 流程。新增对照只改变数据池的随机打乱，不改变 Sequence 的训练 seed schedule：每条线的 8 个 Task 仍使用训练 seed 42–49。

## 共同协议

    全部 9,573 张图
      → 相邻帧去冗余（τ=0.99，window=1，每簇一个代表）
      → 6,914 张代表图
      → 按 split seed 随机划分 1,000 validation + 1,000 test
      → 剩余 4,914 张随机均分为 8 个训练 stage
      → 每个 stage 从 yolov5s.pt 重新 Full Cold 训练 100 epochs

| 实验臂 | 数据打乱 seed | Sequence 训练 seed | 作用 |
|---|---:|---:|---|
| GlobalDedupTau099_PostSplit | 42 | 42–49 | 当前主线 |
| GlobalDedupTau099_PostSplit_RandomS41 | 41 | 42–49 | 随机划分对照 |
| GlobalDedupTau099_PostSplit_RandomS43 | 43 | 42–49 | 随机划分对照 |

三条曲线会各自重新产生 validation/test，因此这是对“完整随机划分协议”的稳健性检查；它不是在同一个 test 集上只改变训练 stage 顺序的单因素实验。比较时必须同时记录各 split 的样本构成和每阶段 mAP 范围。

## 结果位置

- 主线结果：experiment/sequence/GlobalDedupTau099_PostSplit__seq-full-cold__stage0-stage7__yolov5s__s42/
- split seed 41：experiment/sequence/GlobalDedupTau099_PostSplit_RandomS41__seq-full-cold__stage0-stage7__yolov5s__s42/
- split seed 43：experiment/sequence/GlobalDedupTau099_PostSplit_RandomS43__seq-full-cold__stage0-stage7__yolov5s__s42/
- 随机性比较：result/randomization_comparison/
- 各类别 mAP 曲线：result/per_class_learning_curve/
- 对照启动与汇总：scripts/run_randomization_controls.sh

比较脚本只从原始 task_result.json -> metrics.best.test 和 cost.selected_count 取数，并在三条曲线的累计训练图像点不一致时直接失败。当前结果最多支持“在这三个 split seed 下曲线差异较小/存在明显差异”的描述，不能据此宣称对所有随机数或所有数据划分都完全无关。
