# FP/FN AnyLabeling Review

该方法接收一个已完成的 task 结果目录，使用该 task 的 `best.pt` 对指定数据划分重新推理，并构造可直接浏览的错误样本目录。

默认输出结构：

```text
<timestamp>__<task-name>__test__conf-0_25/
├── best.pt
├── best.onnx
├── yolov5_anylabeling.yaml
├── anylabeling_color_snippet.yaml
├── false_positives/
│   ├── <batch>__<original-image-name>       # 图片链接
│   └── <batch>__<original-image-name>.json  # GT/TP/FP 对比框
├── false_negatives/
│   ├── <batch>__<original-image-name>       # 图片链接
│   └── <batch>__<original-image-name>.json  # GT/TP/FP 对比框
├── review_manifest.csv
└── review_summary.json
```

- 一张图只要有至少一个 FP，就链接到 `false_positives/`。
- 一张图只要有至少一个 FN，就链接到 `false_negatives/`。
- 同时包含 FP 和 FN 的图会出现在两个目录中。
- 每个图片链接旁边都有同名 X-AnyLabeling JSON：全部 `GT/<class>` 真实框显示为白色，匹配成功的 `TP/<class>` 预测框显示为绿色，未匹配的 `FP/<class>` 预测框显示为红色。只有白色 GT、没有绿色 TP 覆盖的目标就是 FN，不额外绘制 FN 框。
- 为保持画面简洁，框的 `description`、`group_id` 和 `attributes` 均为空；预测框只保留 `score` 置信度。
- `anylabeling_color_snippet.yaml` 提供上述白/绿/红颜色。可在 X-AnyLabeling 的 Label Manager 中按该文件设置，也可将其中的 `shape_color` 和 `label_colors` 合并到 `.xanylabelingrc`。
- FP/FN 默认采用类别一致且 `IoU >= 0.5` 的一对一匹配，预测置信度阈值为 `0.25`。
- Windows 默认创建硬链接，不移动、不复制图片，也通常不需要管理员权限。硬链接要求输出目录和原图位于同一磁盘卷；跨盘时可使用 `--link-mode symlink`，但 Windows 创建符号链接可能需要开启开发者模式或管理员权限。
- `best.pt` 会复制到输出根目录，并通过固定版本 YOLOv5 的 `export.py` 自动导出为固定输入尺寸、batch=1 的 `best.onnx`。
- `best.onnx` 会经过 `onnx.checker` 和 ONNX Runtime session 加载验证；输入尺寸与输出类别维度不符合预期时脚本会报错。
- `yolov5_anylabeling.yaml` 使用相对路径 `model_path: best.onnx`，可以随整个结果目录移动；类别顺序严格来自实验数据布局。
- `review_manifest.csv` 保存每张错误图的 FP/FN 数量和类别明细。
- `review_summary.json` 记录转换开始/结束时间、总耗时，以及推理、目录与 JSON 构造、ONNX 导出、ONNX 验证各阶段耗时；命令行结束时也会显示总秒数和分钟数。

## 使用方法

在项目根目录运行：

```powershell
conda run --no-capture-output -n yolo-result-analysis python `
  "E:/desktop/project/YOLO Retraining/project_v2/result_analysis/fp_fn_review/build_fp_fn_review.py" `
  "E:/desktop/project/YOLO Retraining/project_v2/runs/tasks/<task-result-directory>" `
  --device 0 --batch 16 --half
```

默认分析测试集，推理和 ONNX 导出尺寸都自动读取 `task.yaml` 中训练时的 `backend.params.imgsz`。ONNX 默认使用 CPU 导出，避免额外占用推理 GPU；可通过 `--export-device` 覆盖。常用参数：

```powershell
# 显式指定输出目录和阈值
python "E:/desktop/project/YOLO Retraining/project_v2/result_analysis/fp_fn_review/build_fp_fn_review.py" `
  "E:/.../runs/tasks/<task-result-directory>" `
  --output-dir "E:/.../result_analysis/fp_fn_review/my-review" `
  --conf-thres 0.25

# 先用少量图片检查环境和目录结构
python "E:/desktop/project/YOLO Retraining/project_v2/result_analysis/fp_fn_review/build_fp_fn_review.py" `
  "E:/.../runs/tasks/<task-result-directory>" `
  --device cpu --limit-images 10
```

从服务器复制回来的 `task.yaml` 可能记录 Linux 绝对路径。脚本会使用其中数据布局文件的文件名，在本项目的 `configs/data/` 下自动寻找 Windows 本地版本；如果未找到，使用 `--data-layout` 显式指定。

训练环境与结果分析环境应相互隔离：`yolo-retraining-v5` 只保留服务器一致的训练依赖，ONNX 与 ONNX Runtime 安装在 `yolo-result-analysis`。在项目根目录可用以下命令从训练环境克隆并构建分析环境：

```powershell
powershell -ExecutionPolicy Bypass -File requirements/bootstrap_analysis.ps1
```

分析环境仍锁定 `numpy==1.26.4`。这是因为项目使用的 Torch 2.2.2 不能与 NumPy 2.x 的数组接口正常协作；不要在训练环境中直接安装可能升级 NumPy 的分析依赖。

直接用 AnyLabeling 分别打开 `false_positives/` 和 `false_negatives/` 目录，就会加载脚本预生成的 GT/TP/FP 对比框，无需再次运行模型。需要尝试其他置信度阈值时，再按 `Ctrl+A` 打开自动标注面板，选择 `Load Custom Model` 并加载输出根目录中的 `yolov5_anylabeling.yaml`。图片链接指向原始图像；JSON 位于 review 目录，不会覆盖原数据目录中的同名标注。
