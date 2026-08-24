# Dataset analysis

The batch argument is shared by analysis scripts and accepts individual batch
numbers and inclusive ranges:

```text
1,2,5-8  -> batch_01, batch_02, batch_05, ..., batch_08
```

Class distribution for selected batches:

```powershell
python -m data_analyse.class_distribution --batches "1,2,5-8"
```

JSON output:

```powershell
python -m data_analyse.class_distribution --batches "1,2,5-8" --format json
```

Use another data location with `--data-root`:

```powershell
python -m data_analyse.class_distribution --data-root E:/path/to/data --batches "1-3"
```

