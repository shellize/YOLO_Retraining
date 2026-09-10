# 全量数据时序去重、人工复核与最终数据池

## 研究目的

本 Study 从 `data/self_improving` 构造冻结的去冗余全量数据池，并进一步生成固定的 train/val/test。它先在全部 20 个物理 batch、9,573 张图片上执行 YOLOv5 多尺度特征时序去重，再对去重后仍然全局高相似的代表图进行人工场景审计。

时序去重先保留 6,039 张图片；人工审阅在最高相似的 200 对中将 13 对标为连续片段或近重复，这些关系形成 10 个连通组。每组优先保留带非空标注的图片，其次保留全局帧号更小者，共排除 12 张，冻结 6,027 张作为唯一的全量划分输入。随后以图片为单位、固定 seed 42 按 8:1:1 分配 train/val/test；分层特征包括类别是否出现、正样本/背景状态和全局帧序号的十分位区间。原始图片和标签不做物理删除。

## 固定协议

- 输入布局：公共 `configs/data/self_improving.yaml` 中的全部图片，包括原有 train、val、test 逻辑组。
- 特征缓存：复用 `data_analyse/dataset_redundancy/results/20260824-182608/embeddings.npz`。运行时必须核对其图片清单与当前布局的 9,573 张图片完全一致。
- 特征定义：固定 YOLOv5 v7.0 `yolov5s.pt` 的 P3/P4/P5 多尺度特征，沿用缓存内记录的提取元数据。
- 时序顺序：解析文件名开头的全局帧号，在全部物理 batch 间统一排序。当前输入的全局帧号唯一，覆盖 `1..9575`，缺少两帧。
- 时序候选：每张图片与全局顺序中向后的 20 张现存图片比较，比较会跨越 `0720_1 → 0720_2` 等 batch 边界。
- 去重阈值：余弦相似度 `tau=0.99`。
- 分组与代表选择：每轮先找覆盖能力最强的普通 greedy 候选；如果其当前邻域含带非空 YOLO 标注的图片，则从这些带标注候选中选择覆盖能力最强者，并以实际选中的代表重新确定本簇成员。这样优先保留正样本，同时保证最终任意两个代表之间不存在本协议定义的超阈值时序边。
- 原始图片和标签始终只读，不复制、不移动、不改名、不删除。

`temporal-window=20` 比项目的相邻帧主协议更宽，可能连接同一相机的短时回访帧。簇预览 HTML 因而是协议的一部分，不能只看统计数字决定去重结果有效。

本 Study 的审计先后排除了两个有缺陷的候选池：第一次试运行沿用公共工具“按父目录分别排序”的定义，最高相似对发现 `0720_1/00500...` 与 `0720_2/00502...` 是跨 batch 的连续画面；第二次虽改为全局顺序，但仍采用“成簇后替换带标注代表”，最高相似对发现帧距仅 2、相似度 `0.99890` 的 `04996/04998` 同时保留，说明事后替换破坏了代表集合的去重约束。两次输出只保留为方法诊断，不作为候选池；正式输出采用上述标注感知 greedy。

## 两层人工审计

1. `result/global_temporal_cluster_preview_label_aware/cluster_preview.html` 展示所有非单例时序簇、真实框、普通候选和实际标注优先代表，也标明跨 batch 簇，用于检查 window=20 是否把不同场景误合并。
2. `result/global_post_dedup_similarity_label_aware/top_similar_pairs.html` 在去重后代表集合中做全局余弦近邻审计，按相似度展示最高的图片对及其全局序列距离。页面把人工结论拆成“连续片段/近重复”“同机位但不同时间”“不同机位/场景”“不确定”，并可导出 CSV。

第二层高相似图片对不受 temporal window 限制。人工导出的状态是二次复核的唯一输入：`near_duplicate` 关系按无向连通组折叠；未审阅以及“同机位但不同时间”“不同机位/场景”“不确定”均保留。复核只生成新的最终 manifest，不修改源数据。

时序去重结果的最高 200 对相似度均高于 `0.99`，但协议内冲突为 0；人工仍识别出 13 条较长距离的连续片段或近重复关系，并将其折叠为 10 个组。人工复核表明，其余高相似图片基本属于同一视角下的不同时间，而不是应继续折叠的连续帧。当前评估目标是同一业务环境和相机分布内的随机泛化，因此最终切分不按 batch、时间段或机位成组，而采用图片级分层随机划分。同一视角跨集合是本协议有意保留的目标分布，不解释为场景外泛化能力。

## 候选划分与已发现的时序漏洞

- 输入：`global_order_window20_tau0p990_label_aware_reviewed_final/manifest.txt` 中的 6,027 张图片。
- 比例与数量：train/val/test 为 8:1:1，目标数量分别为 4,821、603、603。
- 随机性：固定 split seed 42；训练 seed 是另一变量，不由本划分定义。
- 分层约束：同时平衡多标签类别出现、正样本/背景比例，并让三个集合覆盖全局帧序号的十个等频区间。
- 使用边界：该候选划分在相邻保留帧审计完成前不用于正式训练或结论。
- 后续学习曲线：只从 train 构造嵌套训练子集，val/test 始终保持不变。
- test 难度审阅：只展示 test 中带有效标注的正样本；背景图片不参与人工筛除。人工明确标记为 `difficult` 的图片将在后续 filtered test 中排除，未审阅、`normal` 和 `unsure` 均默认保留。完整 test 始终保留不变。

在 test 难度审阅中发现 `0720_2/00550...` 与 `00551...` 是明显的连续事件帧。两者虽在 window=20 内，但 YOLO 特征余弦相似度仅为 `0.967472`，低于 `tau=0.99`，因此均作为单例保留；非单例簇审阅页不会展示它们。全局高相似审阅页又只展示相似度最高的 200 对，其最低相似度约为 `0.996158`，“按帧距排序”只对这 200 对重排，因此也无法暴露该图片对。

进一步审计发现最终池仍有 4,036 对前缀序号差 1 的相邻保留图片，其中候选随机划分有 1,393 对跨 split；这些相邻关系形成 806 条长度大于 1 的序号相邻候选链，共涉及 4,835 张图片，最长候选链为 125 张。序号相邻链只用于展示上下文，不预设为真实连续事件。另有 165 对文件名来源时间戳完全相同，其中 53 对跨 split。同时间戳双图只是最容易确认的漏洞，不是最终分组单位。这说明 `tau=0.99` 只完成了近乎完全重复压缩，不能独立承担连续事件隔离。`retained_temporal_pair_audit` 对相邻边逐一记录人工结论，后续把 `near_duplicate` 边按连通分量合并，从而覆盖两张、三张及更长连续事件链。`random_stratified_s42_8_1_1` 因而保留为候选和问题证据；正式 split 必须等待该审计确定事件绑定规则后重建。test 困难样本审阅也随之暂停。

## 输出边界

```text
runs/studies/0910_all_data_window20_tau099_test_construction/
├── config/protocol.yaml
├── experiment/variants/global_order_window20_tau0p990_label_aware_independent_representative/
│   ├── manifest.txt
│   ├── clusters.csv
│   ├── protocol.json
│   └── summary.json
├── experiment/variants/global_order_window20_tau0p990_label_aware_reviewed_final/
│   ├── manifest.txt
│   ├── manual_review.csv
│   ├── review_decisions.csv
│   ├── protocol.json
│   └── summary.json
├── experiment/variants/random_stratified_s42_8_1_1/
│   ├── manifests/{train,val,test}.txt
│   ├── layout.yaml
│   ├── assignments.csv
│   ├── protocol.json
│   └── summary.json
├── result/global_temporal_cluster_preview_label_aware/
├── result/global_post_dedup_similarity_label_aware/
├── result/test_positive_difficulty_review/
├── result/retained_temporal_pair_audit/
├── logs/
└── scripts/
```

`global_order_window20_tau0p990_label_aware_reviewed_final/manifest.txt` 是冻结的最终全量数据池。`random_stratified_s42_8_1_1/` 是当前固定划分，其中三个 manifest 是 train/val/test 的数据身份来源，`layout.yaml` 可直接交给训练框架。`manual_review.csv` 保存原始人工判断，`review_decisions.csv` 保存每个近重复连通组的保留与排除结果。两个 HTML 是审计界面，不作为最终数据身份来源。
