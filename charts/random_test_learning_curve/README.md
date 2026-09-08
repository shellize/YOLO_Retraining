# Random-test Full Cold learning curve

This directory contains the plots for the completed `random_frame_s42` Full
Cold sequence. The original data layout was randomly reassigned into train,
validation, and test groups while preserving the original group sizes.

The plots use each stage's `best` checkpoint evaluated on the randomly assigned
test split:

- `map_learning_curve.png`: test mAP50 and mAP50-95;
- `per_class_ap50_learning_curve.png`: test AP50 for each class;
- `per_class_ap_learning_curve.png`: test AP50-95 for each class;
- `metrics.csv`: stage-level values, class AP50/AP50-95, and source `task_result.json` paths;
- `generate_report.py`: reproducible report generator.

The class AP50 values are read from each YOLOv5 `evaluation.log`: the older
JSON bridge retained only the class AP50-95 values even though the evaluator
printed both columns.

Source sequence:
`runs/studies/0824_fullcold_redundancy_and_random_test/experiment/sequence/random_frame_s42__seq-full-cold__stage0-stage3__yolov5s__s42`
