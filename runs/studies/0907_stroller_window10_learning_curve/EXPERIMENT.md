# stroller 冗余审计与 window=10 学习曲线 Study

## 目的

本 Study 统一保存 `stroller` 的冗余审计、最终采用的数据协议和学习曲线实验。先对人工校验后的 898 张图片做相似度审计，再采用 `τ=0.99、temporal-window=10` 的代表图保留集进行随机划分和 Full Cold learning-curve 实验。`stroller_raw` 仅保留为未经人工校验的原始数据，不进入本 Study。

## 冗余审计协议

- 输入：`data/stroller`，898 张图片、898 个非空 YOLO TXT 标签，标签仅含 `stroller`（类别 ID `1`）。`data/stroller_raw` 不作为实验输入。
- 特征：固定 YOLOv5 v7.0、COCO 预训练 `yolov5s.pt`，P3/P4/P5 特征做 L2 归一化。
- 邻近关系：同一 `images/` 目录内按文件名排序，每张图向后检查指定 temporal window。
- `τ=0.98/0.99` 和多个 window 仅用于数据审计；本 Study 最终采用 `window=10、τ=0.99`。
- 保留策略：每个相似簇保留一个代表图，不修改 `data/stroller` 原始数据。

## 最终学习曲线协议

- 去冗余后保留 801 张图片。
- 每个 split seed 独立随机划分：`train:test:val = 641:80:80`，约为 8:1:1。
- split seeds：41、42、43。
- 每个训练集再分为 8 个 stage：`81,80,80,80,80,80,80,80`。
- 每个 stage 使用截至当前 stage 的全部已见训练图片，并从 `yolov5s.pt` 重新 Full Cold 训练。
- 默认模型协议沿用公共 learning-curve 模板：YOLOv5s、100 epochs、640 输入尺寸、batch 64；启动前可在服务器脚本中调整 GPU 和环境变量。
- 三条曲线的训练 seed schedule 固定为 42–49；变化因素是数据分配 seed。

## 结果和数据位置

- 原始冗余结果：`result/adjacent_redundancy/`
- window 扫描：`result/window_sweep_tau_0p990/`
- 最终 window=10 manifest：`result/post_dedup_window10_tau_0p990/manifest.txt`
- window=10 非单例簇预览：`result/window_sweep_tau_0p990/window_10_non_singleton/`
- 三组训练布局：`experiment/variants/window10_tau0p990_split_s41/`、`s42/`、`s43/`
- Sequence 原始产物：`experiment/sequence/`
- 学习曲线汇总：`result/learning_curve/`

## 启动边界

当前 Study 已完成数据布局、配置和脚本预检，但尚未启动 stroller 训练。服务器端使用 `scripts/launch_learning_curves_when_gpu0_idle.sh` 轮询 GPU0；默认只准备和验证，连续两次空闲后按顺序运行全局去冗余 Study 的待运行 Sequence，再运行本 Study。训练必须显式传入 `--start-training`。

三条随机划分曲线只支持描述这三个 split seed 下的差异，不能外推为所有随机划分都稳定。随机划分也不是场景级隔离协议，跨时间的高相似图片可能进入不同 split。
