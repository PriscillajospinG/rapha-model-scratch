# Final Training Report — Lower Limb CTR-GCN

## Dataset
| Split | Samples |
|-------|---------|
| Train | 279 |
| Val   | 75   |
| Test  | 96  |
| Total | 450 |

Split is group-aware: a real video and any augmented clips derived from it
always land in the same split, so the test set contains no near-duplicates
of training data.

## Model Performance
| Metric | Value |
|--------|-------|
| Best Top-1 Accuracy (test, untouched during training) | 0.4375 |
| Best Val Accuracy (used for checkpoint selection) | 0.5733 |
| Macro F1 (test) | 0.4397 |
| Top-3 Accuracy (test) | 0.6979 |

**Status: Below target — see recommendations below ⚠️**

## Exported Files
| File | Location |
|------|----------|
| PyTorch weights | `models/lower_limb\best_model.pth` (4.61 MB) |
| ONNX model | `models/lower_limb\best_model.onnx` (0.09 MB) |
| Deployment metadata | `models/lower_limb\deployment_metadata.json` |

## Recommendations for Improvement
1. **Increase dataset size** — collect more real videos per class, especially for weaker classes (toes, knee, leg_raise, heel_slide)
2. **Tune hyperparameters** — try lr=0.0005, or a shallower/wider model given the small dataset
3. **Check confusion matrix** — identify which class pairs confuse the model most (previously knee/quadriceps, hip/leg_raise)
4. **Review low-visibility joints** — MediaPipe visibility scores for toe/heel landmarks are often lower quality on short clips; consider filtering or flagging low-confidence frames
