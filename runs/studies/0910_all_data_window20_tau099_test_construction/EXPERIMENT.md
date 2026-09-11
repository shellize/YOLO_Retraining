# 全量数据去重、事件隔离与测试集构造

## 研究目的

既有实验中，按 batch 取 test 会因时间分布偏移而表现很差，图片级随机取 test 又会因连续事件泄露而表现过好，因此当前主要问题不是继续调整训练方法，而是建立可信的评估数据协议。

本 Study 从 `data/self_improving` 的 20 个物理 batch、9,573 张图片出发，先压缩近乎完全重复的临近图片，再人工排除未被整图相似度捕获的连续正样本，最后使用目标级相似度把同一目标的持续出现绑定为不可跨 split 的图片组。原始图片和标签始终只读，不复制、不移动、不改名、不删除。

当前结果是 5,763 张最终图片，以及 225 个不可跨 split 的图片级 group，共约束 871 张图片。在此基础上已经冻结 group-aware 的 8:1:1 随机分层划分；旧的 6,027 张图片级随机划分仅保留为问题证据，不再作为正式训练、验证或测试数据。

## 固定数据与时序协议

- 输入布局：公共 `configs/data/self_improving.yaml` 中的全部图片，包括原有 train、val、test 逻辑组。
- 特征缓存：复用 `data_analyse/dataset_redundancy/results/20260824-182608/embeddings.npz`，其图片清单必须与当前布局的 9,573 张图片一致。
- 整图特征：固定 YOLOv5 v7.0 `yolov5s.pt` 的 P3/P4/P5 多尺度特征。
- 时序顺序：解析文件名开头的全局帧号，在所有 batch 间统一排序；batch 只是物理切分，不作为事件边界。当前帧号唯一，覆盖 `1..9575`，缺少两帧。
- “去重后距离”表示图片在完整的去重后 manifest 中的位置差；背景图片即使不参与目标匹配，也仍占据序列位置。
- 原始数据始终只读；人工判断只生成新的 manifest 或 group 约束。

## 实验过程与方法转变

### 1. 按相似度对临近图片去重

最初假设是连续采集图片通常高度相似，因此在全局帧序列中比较每张图片之后的 20 张现存图片，并以余弦相似度 `tau=0.99` 建立时序关系。代表选择采用标注感知 greedy：如果候选邻域中存在带非空标注的图片，就优先从正样本中选择覆盖能力最强者，并按实际代表重新确定簇成员。

这一阶段发现：

- 相似度很高不一定表示重复，也可能是同一机位下的不同时刻。
- 相似度不高也可能是重复或连续事件帧，因为目标、遮挡和背景变化会降低整图相似度。
- 图片在时间上临近，是判断连续事件的重要条件；只做全局高相似检索不能替代时序关系。
- batch 之间也可能连续，因此必须使用跨 batch 的全局帧顺序。

审计还排除了两个有缺陷的候选池：第一次按父目录分别排序，漏掉了跨 batch 的连续图片；第二次先成簇再替换带标注代表，使帧距 2、相似度 `0.99890` 的图片同时保留，破坏了代表集合的去重约束。正式实现改为上述标注感知 greedy。

自动时序去重将 9,573 张图片缩减为 6,039 张。随后审阅去重后全局相似度最高的 200 对图片，人工确认 13 条近重复边、10 个连通组，再排除 12 张，得到 6,027 张候选数据。该审阅同时确认：其余大量高相似图片只是同机位的不同时刻，不应继续删除。

### 2. 人工审计临近且标注位置相似的正样本

基于 6,027 张图片曾构造一版 seed 42、比例 8:1:1 的图片级分层随机划分，同时平衡类别、正样本/背景和全局时间十分位。它得到 train/val/test = 4,821/603/603，但只作为候选划分。

在 test 正样本难度审阅中发现 `0720_2/00550...` 与 `00551...` 是明显连续帧，而整图余弦相似度只有 `0.967472`：低于 `tau=0.99`，也低于最高 200 对审阅页约 `0.996158` 的展示下限。进一步统计发现仍有 4,036 对帧号差 1 的保留图片，其中 1,393 对跨越候选 split；另有 165 对来源时间戳相同，其中 53 对跨 split。这证明 `tau=0.99` 只能压缩近乎完全重复，不能承担连续事件隔离。

真正严重的评估泄露是同一个带标注目标同时进入 train 与 val/test，因此人工审计不再关注大量纯背景，只筛选双方都有有效标注的临近图片。为了减少审阅量，进一步要求两图总框数相同，并将 YOLO 框视为无序集合，用匈牙利算法一一匹配，按照平均 `中心坐标 L2 距离 + 0.5 × 宽高 L2 距离` 从小到大展示。类别不参与匹配，以免漏掉画面重复但标注类别不一致的图片。

人工双图审计分两轮：

1. 原始帧号差 1、双方正样本、总框数相同：1,114 对候选，确认 211 条重复边。
2. 去重后保留序列中相邻、原始帧号差 2～20、双方正样本、总框数相同且整图相似度大于 `0.96`：96 对候选，确认 53 条重复边。

两轮共确认 264 条重复边，形成 183 个连通分量；每个分量保留全局帧号最小的图片，最终排除 264 张，得到 5,763 张最终数据池 `global_order_window20_tau0p990_label_aware_pair_reviewed_final`。

这一阶段又暴露出整图方法的上限：地铁场景中，同一个人可能在站台停留数分钟，目标位置和外观基本不动，但背景人群持续变化。即使属于同一目标事件，整图相似度也可能较低，双图近重复审计无法完整描述这种长时间持续出现。

### 3. 使用目标相似度形成轨迹和 split group

为处理同一目标在背景变化下的持续出现，判断单位从“整张图片是否重复”改成“标注框中的目标是否属于同一轨迹”。`target_track_candidate_audit` 在完整的 5,763 张 manifest 顺序中，只匹配前后距离不超过 20 的同类别目标，并综合：

- 紧贴标注框的目标外观；
- 少量外扩上下文；
- 框中心和大小变化；
- HSV 直方图与 HOG 描述；
- 当前目标同时对轨迹原型和最近节点的匹配程度。

粗筛采用偏低阈值以优先保证召回，轨迹得分取轨迹内最低连接得分并由高到低排序。审阅页只展示目标裁剪序列，人工判断 `same_track`、`not_same_track` 或 `unsure`。`same_track` 整体形成目标事件；未审阅和 `not_same_track` 默认不绑定；`unsure` 按手工清单拆分，单图条目只从错误轨迹中剥离，不产生跨图片约束。

第一轮处理 4,176 个标注目标，生成 653 条候选轨迹。人工审阅和拆分后确认 248 条目标事件，形成 192 个图片级 group，共约束 779 张图片，最大 group 为 21 张。

第二轮只处理尚未绑定轨迹的剩余图片，同时将紧框目标外观、外扩上下文和框几何的权重调整为 `0.80/0.10/0.10`。该轮处理 2,923 个目标，生成 372 条候选轨迹；人工审阅后新增 34 条目标事件、33 个图片级 group，共约束 92 张图片。

两轮结果统一重编号，并按共享图片关系合并，最终得到：

- 282 条人工确认的目标事件；
- 225 个不可跨 split 的图片级 group；
- 871 张被 group 约束的图片；
- 最大 group 为 21 张。

`target_track_groups_two_pass_final/image_split_groups.csv` 是后续正式划分的唯一轨迹约束来源。目标级轨迹阶段不再删除图片，只规定哪些图片必须进入同一个 split。

## 当前结论与后续评估边界

当前方法转变可以概括为：

```text
整图相似度去重
  → 相似度不能准确等价于重复
标注感知的临近双图审计
  → 同一目标可能长期停留且背景持续变化
目标级轨迹识别与人工分组
  → 得到不可跨 split 的事件约束
```

最终划分 `group_stratified_s42_8_1_1` 从 5,763 张数据池出发，把 225 个 group 和其余独立图片视为不可拆分的分配单元。硬约束包括 group 完整、全部图片恰好分配一次和图片数量精确为 8:1:1；软目标同时平衡类别正样本图片数、类别实例数、正负样本和十个全局时间区间。为避免 val/test 被单一轨迹稀释，group 图片总量也按约 8:1:1 分配，且 10 张及以上的大 group 优先进入 train。

最终 train/val/test 图片数为 `4,611/576/576`，正样本数为 `2,371/296/296`。225 个 group 按 `199/13/13` 分配，group 图片数为 `697/87/87`；16 个大小不低于 10 的 group 全部进入 train，val 和 test 的最大 group 均为 9 张。三个 split 对最终 manifest 完整覆盖、互不重叠，且同一 group 不跨 split。旧的 `random_stratified_s42_8_1_1` 不再用于正式训练或结论。

正式 split 冻结后，只审阅 test 中的正样本困难度，并保留两套评价：完整 test 表示真实目标分布，filtered test 排除人工明确确认的极困难或不可判定标注。学习曲线用于检查评估是否稳定，但不能反向调整 test 直到曲线符合预期，以免对测试集产生人为过拟合。

当前证据只约束人工确认的同一轨迹。未审阅候选默认独立，目标匹配窗口也有限，因此不能声称所有潜在身份泄露已经被完全消除。

## 主要产物

```text
runs/studies/0910_all_data_window20_tau099_test_construction/
├── config/protocol.yaml
├── config/target_track_manual_overrides.yaml
├── config/target_track_second_pass_manual_overrides.yaml
├── experiment/variants/
│   ├── global_order_window20_tau0p990_label_aware_independent_representative/
│   ├── global_order_window20_tau0p990_label_aware_reviewed_final/
│   ├── global_order_window20_tau0p990_label_aware_pair_reviewed_final/
│   ├── random_stratified_s42_8_1_1/                    # 已作废的图片级候选划分
│   ├── group_stratified_s42_8_1_1/                     # 正式 group-aware 划分
│   ├── target_track_groups_reviewed_final/
│   ├── target_track_second_pass_groups_reviewed_final/
│   └── target_track_groups_two_pass_final/             # 最终轨迹约束
├── result/
│   ├── global_temporal_cluster_preview_label_aware/
│   ├── global_post_dedup_similarity_label_aware/
│   ├── test_positive_difficulty_review/                 # 旧候选 test 审阅，暂停使用
│   ├── retained_temporal_pair_audit/
│   ├── positive_box_layout_pair_audit/
│   ├── positive_box_layout_nonconsecutive_pair_audit/
│   ├── final_positive_tau095_window10_cluster_audit/
│   ├── target_track_candidate_audit/
│   └── target_track_second_pass_audit/
└── scripts/
```

HTML 和图表只用于人工审计，不作为数据身份来源。最终图片身份由 `global_order_window20_tau0p990_label_aware_pair_reviewed_final/manifest.txt` 给出，最终事件约束由 `target_track_groups_two_pass_final/image_split_groups.csv` 给出。
