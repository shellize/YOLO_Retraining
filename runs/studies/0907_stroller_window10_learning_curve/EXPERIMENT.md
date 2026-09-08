# stroller 冗余审计与学习曲线 Study

## 目的

本 Study 统一保存人工校验后的 `stroller` 数据集冗余审计、最终采用的数据协议和 learning-curve 实验。已有三组随机切分结果保留不变；新增一个尽量保持去冗余后 manifest 原顺序的顺序切分对照。`stroller_raw` 仍是未经人工校验的数据，不进入本 Study。

## 冗余审计协议

- 输入：`data/stroller`，898 张图片、898 个非空 YOLO TXT 标签，标签仅含 `stroller`（类别 ID `1`）。
- 特征：固定 YOLOv5 v7.0、COCO 预训练 `yolov5s.pt`，P3/P4/P5 特征做 L2 归一化。
- 邻近关系：同一 `images/` 目录内按文件名排序，每张图向后检查指定 temporal window。
- `τ=0.98/0.99` 和多个 window 仅用于数据审计；最终采用 `window=10、τ=0.99`。
- 保留策略：每个相似簇保留一个代表图，不修改 `data/stroller` 原始数据。

## 学习曲线协议

- 去冗余后保留 801 张图片。
- 随机实验分别按 split seed 41、42、43 划分 `train:test:val = 641:80:80`，每个训练集再分为 8 个 stage：`81,80,80,80,80,80,80,80`。
- 新增顺序对照使用同一 801 张图片和同一比例，直接沿用去冗余输出 manifest 的 canonical path 顺序；按现有脚本的区间顺序取 train 80%、test 10%、val 10%，训练区间内部保持原顺序分成 8 个 stage。
- 每个 stage 使用截至当前 stage 的全部已见训练图片，并从 `yolov5s.pt` 重新 Full Cold 训练。
- 默认模型协议沿用公共 learning-curve 模板：YOLOv5s、100 epochs、640 输入尺寸、batch 64。
- 随机切分和顺序对照都使用训练 seed schedule 42–49。YOLOv5 训练 epoch 内仍保留默认 DataLoader shuffle；本对照的变量是切分前后的样本分配顺序。

| 实验臂 | Sequence |
|---|---|
| 随机 split seed 41 | `StrollerWindow10Tau099_SplitS41` |
| 随机 split seed 42 | `StrollerWindow10Tau099_SplitS42` |
| 随机 split seed 43 | `StrollerWindow10Tau099_SplitS43` |
| 顺序切分对照 | `StrollerWindow10Tau099_Ordered` |

## 结果和数据位置

- 原始冗余结果：`result/adjacent_redundancy/`
- window 扫描：`result/window_sweep_tau_0p990/`
- 最终 window=10 manifest：`result/post_dedup_window10_tau_0p990/manifest.txt`
- 三组随机布局：`experiment/variants/window10_tau0p990_split_s41/`、`s42/`、`s43/`
- 顺序布局：`experiment/variants/window10_tau0p990_ordered/`
- Sequence 原始产物：`experiment/sequence/`
- 随机曲线：`result/learning_curve/learning_curve_by_split_seed.png`
- 顺序对照综合曲线：`result/learning_curve/learning_curve_with_ordered_control.png`

## 启动边界

本次只加入顺序布局、配置、启动和结果汇总代码，未在本地启动新的训练。服务器端的 `scripts/run_learning_curve.sh` 会在三条随机曲线之后运行顺序对照，并在 8 个 Task 全部完成后生成综合比较图。已有 Task/Sequence 原始结果保持只读，不覆盖不完整输出。

顺序切分不是场景级隔离协议的完整实现；它只是对当前 manifest 顺序的敏感性对照。结果解释必须同时记录数据区间和 split 构成。
