# 全量数据时序去重与 test 候选池审计

## 研究目的

本 Study 为后续 test 构造准备一个去冗余候选池。它先在 `data/self_improving` 的全部 20 个物理 batch、9,573 张图片上执行 YOLOv5 多尺度特征时序去重，再对去重后仍然全局高相似的代表图进行人工场景审计。

本阶段不直接生成最终 train/val/test 划分。人工审阅完成前，去重 manifest 只能称为 test 候选池，不能作为已经冻结的 benchmark。

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

第二层高相似图片对不受 temporal window 限制。它用于验证“高特征相似度不必然等于连续重复帧”，审阅结果不会自动触发二次删除。

正式候选池的最高 200 对相似度均高于 `0.99`，但其最小全局序列距离为 21，协议内冲突为 0。对排名前三的初步人工检查显示，它们是相同固定机位在不同时间拍到的不同人员/事件，并非连续帧；因此当前结果支持“时序去重已生效”，却不支持“剩余高相似样本属于完全不同物理场景”。如果 test 需要衡量跨机位或跨场景泛化，后续切分还必须使用机位/场景 group-disjoint 协议，不能在这 6,039 张代表中直接随机抽图。

## 输出边界

```text
runs/studies/0910_all_data_window20_tau099_test_construction/
├── config/protocol.yaml
├── experiment/variants/global_order_window20_tau0p990_label_aware_independent_representative/
│   ├── manifest.txt
│   ├── clusters.csv
│   ├── protocol.json
│   └── summary.json
├── result/global_temporal_cluster_preview_label_aware/
├── result/global_post_dedup_similarity_label_aware/
├── logs/
└── scripts/
```

`experiment/variants/` 的 manifest 与协议是派生 test 候选池；两个 HTML 和人工导出的 CSV 是审计材料。最终 test 的 group-disjoint 划分需要在人工审核后另行冻结，并记录版本和清单哈希。
