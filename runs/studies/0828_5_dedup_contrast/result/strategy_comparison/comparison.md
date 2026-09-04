# Balance tau=0.90 label-contrast comparison

所有行均使用 balance test/val、YOLOv5s、640 分辨率和 full-cold 初始化。
除 Sequence 参考行外，其余为 seed 42 的单 Task；Sequence 行取现有四阶段 full-cold 实验的最终 stage（该 stage 的 seed 为 45）。
`best` 指验证集选择出的 best checkpoint 在 test 上的指标。

| 实验 | 协议 | 训练图数 | imgsz | 配置 epoch | patience | 实际 epoch | best mAP50-95 | best mAP50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 全量原始 baseline（640，配置 300 epoch） | single Task | 7573 | 640 | 300 | 100 | 210 | 0.2385 | 0.5032 |
| 全量原始 baseline（Sequence 最终 stage，100 epoch） | Sequence final stage | 7573 | 640 | 100 | 100 | 100 | 0.2369 | 0.5012 |
| 普通去冗余 | single Task | 4058 | 640 | 300 | 100 | 218 | 0.2008 | 0.4106 |
| 释放含标注簇、纯背景簇留代表 | single Task | 5866 | 640 | 300 | 100 | 221 | 0.2148 | 0.4641 |
| 释放含标注簇、删除纯背景簇 | single Task | 3752 | 640 | 300 | 100 | 156 | 0.2087 | 0.4433 |
| 有标注优先单代表 | single Task | 4058 | 640 | 300 | 100 | 205 | 0.1938 | 0.4416 |

注意：Sequence 参考行来自 `runs/sequences/Balance_Test__seq-full-cold__stage0-stage3__yolov5s__s42` 的最终 stage。它每个 stage 冷启动训练 100 epoch，最终 stage 使用累计 7573 张训练图；四个 tau=0.90 实验则是单 Task、配置 300 epoch/patience=100。因此该行是同分辨率的 100-epoch 全量参考，但不是与四个 tau=0.90 单 Task 完全相同的训练协议。

结果路径和原始 Task 配置见 `comparison.csv`。
