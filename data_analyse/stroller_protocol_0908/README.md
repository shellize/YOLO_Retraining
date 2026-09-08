# stroller 0908 数据协议

本目录保存本次 stroller 困难样本实验所需的可复用数据分析入口。物理数据不进入 Git。

服务器上的两个上传源为：

- `data/stroller_protocol_0908/stroller_raw_source`：从原始图片中筛出的全部 stroller 图片；
- `data/stroller_protocol_0908/stroller_easy_source`：`data/stroller` 对应的困难样本筛选结果。

两者都先独立执行 `tau=0.99、temporal-window=10` 去冗余，再分别用于后续 Study。由于当前服务器磁盘空间有限，上传源采用了带日期的版本目录，没有覆盖已有 `data/stroller`。

`run_dedup.sh` 只做特征提取、相似度审计和代表样本 manifest 构造，不启动检测器训练。它的输出是：

```text
results/difficult/variants/dedup_tau_0p990/manifests/stroller_raw.txt
results/easy/variants/dedup_tau_0p990/manifests/stroller_easy_source.txt
```

两个 Study 的准备脚本只引用这两个 manifest，不复制图片、不修改源数据。比较实验明确不包含 m3：第二个 Study 只训练 m1（difficult）和 m2（easy）。

## 服务器运行

```bash
bash data_analyse/stroller_protocol_0908/run_dedup.sh
```

脚本会拒绝覆盖已有的非空结果目录。去冗余完成后，再分别运行两个 Study 的默认准备模式；只有显式传入 `--start-training` 才会启动训练。
