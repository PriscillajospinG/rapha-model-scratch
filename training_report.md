# Final Training Report — Lower Limb CTR-GCN

## Dataset
| Split | Samples |
|-------|---------|
| Train | 360 |
| Test  | 90  |
| Total | 450 |

## Model Performance
| Metric | Value |
|--------|-------|
| Best Top-1 Accuracy | 0.6778 |
| Macro F1 | 0.6800 |
| Top-3 Accuracy | 0.8222 |

**Status: Below target — see recommendations below ⚠️**

## Exported Files
| File | Location |
|------|----------|
| PyTorch weights | `models/lower_limb\best_model.pth` (4.60 MB) |
| ONNX model | `models/lower_limb\best_model.onnx` (0.09 MB) |
| Deployment metadata | `models/lower_limb\deployment_metadata.json` |

## Recommendations for Improvement
1. **Increase dataset size** — collect more real videos per class
2. **Add spatial augmentation** — slight joint jittering during training
3. **Tune hyperparameters** — try lr=0.0005, larger model depth
4. **Check confusion matrix** — identify which class pairs confuse the model most
