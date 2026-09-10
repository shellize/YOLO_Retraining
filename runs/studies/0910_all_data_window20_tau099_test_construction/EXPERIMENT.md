# 全量数据时序去重与 test 候选池审计

## 研究目的

本 Study 为后续 test 构造准备一个去冗余候选池。它先在 `data/self_improving` 的全部 20 个物理 batch、9,573 张图片上执行与既有连续帧研究一致的 YOLOv5 多尺度特征去重，再对去重后仍然全局高相似的代表图进行人工场景审计。

本阶段不直接生成最终 train/val/test 划分。人工审阅完成前，去重 manifest 只能称为 test 候选池，不能作为已经冻结的 benchmark。

## 固定协议

- 输入布局：公共 `configs/data/self_improving.yaml` 中的全部图片，包括原有 train、val、test 逻辑组。
- 特征缓存：复用 `data_analyse/dataset_redundancy/results/20260824-182608/embeddings.npz`。运行时必须核对其图片清单与当前布局的 9,573 张图片完全一致。
- 特征定义：固定 YOLOv5 v7.0 `yolov5s.pt` 的 P3/P4/P5 多尺度特征，沿用缓存内记录的提取元数据。
- 时序候选：同一物理 batch 目录内按文件名排序，每张图片只与向后 20 张图片比较。
- 去重阈值：余弦相似度 `tau=0.99`。
- 分组：沿用项目已有的 greedy representative grouping；每簇保留一个代表。
- 代表选择：如果簇内存在至少一张带非空 YOLO 标注的图片，最终代表必须优先从带标注成员中选择；否则保留普通代表。
- 原始图片和标签始终只读，不复制、不移动、不改名、不删除。

`temporal-window=20` 比项目的相邻帧主协议更宽，可能连接同一相机的周期性回访帧。簇预览 HTML 因而是协议的一部分，不能只看统计数字决定去重结果有效。

## 两层人工审计

1. `result/temporal_cluster_preview/cluster_preview.html` 展示所有非单例时序簇、真实框、普通代表和标注优先代表，用于检查 window=20 是否把不同场景误合并。
2. `result/post_dedup_similarity/top_similar_pairs.html` 在去重后代表集合中做全局余弦近邻审计，按相似度展示最高的图片对。页面可人工标记“不同场景”“仍是同场景/重复”“不确定”，并导出 CSV。

第二层高相似图片对不受 temporal window 限制。它用于验证“高特征相似度不必然等于连续重复帧”，审阅结果不会自动触发二次删除。

## 输出边界

```text
runs/studies/0910_all_data_window20_tau099_test_construction/
├── config/protocol.yaml
├── experiment/variants/window20_tau0p990_positive_representative/
│   ├── manifest.txt
│   ├── clusters.csv
│   ├── protocol.json
│   └── summary.json
├── result/temporal_cluster_preview/
├── result/post_dedup_similarity/
├── logs/
└── scripts/
```

`experiment/variants/` 的 manifest 与协议是派生 test 候选池；两个 HTML 和人工导出的 CSV 是审计材料。最终 test 的 group-disjoint 划分需要在人工审核后另行冻结，并记录版本和清单哈希。
