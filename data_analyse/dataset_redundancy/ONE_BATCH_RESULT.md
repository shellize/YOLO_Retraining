# One-batch redundancy result: 0720_14

Scope: only physical batch `images/0720_14`, as requested. No full-dataset
embedding or similarity run was performed.

## Input

- images: 500
- background images: 349 (69.8%)
- annotated instances: large luggage 176, stroller 8, wheelchair 1, flatbed truck 1
- weights: the original COCO-pretrained `yolov5s.pt`
- features: YOLOv5s layers 17/20/23, spatial-pyramid pooling 4x4/2x2/1x1
- temporal comparison window: 20 neighboring filenames in the same batch

## Threshold sweep

| Cosine threshold | Representatives | Retained | Candidate redundant images | Largest group |
| ---: | ---: | ---: | ---: | ---: |
| 0.900 | 60 | 12.0% | 440 | 40 |
| 0.950 | 78 | 15.6% | 422 | 21 |
| 0.970 | 116 | 23.2% | 384 | 21 |
| 0.990 | 204 | 40.8% | 296 | 17 |
| 0.995 | 261 | 52.2% | 239 | 17 |
| 0.997 | 321 | 64.2% | 179 | 15 |
| 0.999 | 383 | 76.6% | 117 | 3 |

The first global-average-pooling smoke was rejected after boundary images near
0.99 still showed visibly different crowd states. The spatial-pyramid result
above is the corrected run. Boundary galleries show that 0.90-0.99 remains too
permissive for a strict duplicate claim. `0.999` is the current conservative
candidate; `0.995` and `0.997` remain useful sensitivity settings.

This is evidence of substantial sequential redundancy in one batch, but it is
not yet evidence that the retained subset preserves detector AP. Each generated
deduplicated layout therefore has three equal-size random controls. The next
proof step is training the conservative and random-matched subsets under both
equal-epoch and equal-image-read budgets.

Generated local artifacts are under:

```text
data_analyse/dataset_redundancy/results/0720_14_smoke/analysis_spatial/
```
