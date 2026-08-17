# Result Analysis

每个直接子目录代表一种分析方法。方法目录中放置该方法的 Python 分析脚本，以及一个或多个独立的结果子目录。

当前方法：

```text
result_analysis/
├── train_test_overfitting/
│   ├── analyze_train_test_overfitting.py
│   ├── README.md
│   └── <analysis-result-directory>/
└── fp_fn_review/
    ├── build_fp_fn_review.py
    ├── README.md
    └── <review-result-directory>/
```

新增分析方法时，在本目录下新增英文命名的方法目录，不要把不同实验名称直接作为本层目录。不同数据集、权重或参数设置的结果应放在对应方法目录下的独立结果子目录中。

结果分析统一使用独立 Conda 环境 `yolo-result-analysis`，避免 ONNX 等分析依赖改变服务器一致的训练环境 `yolo-retraining-v5`。首次构建时在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File requirements/bootstrap_analysis.ps1
```
