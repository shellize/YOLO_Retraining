# Result Analysis

每个直接子目录代表一种分析方法。方法目录中放置该方法的 Python 分析脚本，以及一个或多个独立的结果子目录。

当前方法：

```text
result_analysis/
└── train_test_overfitting/
    ├── analyze_train_test_overfitting.py
    ├── README.md
    └── <analysis-result-directory>/
```

新增分析方法时，在本目录下新增英文命名的方法目录，不要把不同实验名称直接作为本层目录。不同数据集、权重或参数设置的结果应放在对应方法目录下的独立结果子目录中。
