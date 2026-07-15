import os
import json
import torch
import torch.nn as nn
from phase_6_7_train import CTRGCN, CLASSES

BASE_DIR = "datasets/lower_limb"
MODEL_DIR = "models/lower_limb"
EVAL_DIR = "evaluation/lower_limb"
EXPORT_DIR = "models/lower_limb"

def export_and_report():
    print("\n--- Starting Phase 8 & 9: Export and Report ---")
    
    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print("Error: best_model.pth not found. Train the model first.")
        return
        
    device = torch.device("cpu")
    model = CTRGCN(num_class=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Export to ONNX
    dummy_input = torch.randn(1, 4, 300, 10, 1)
    onnx_path = os.path.join(EXPORT_DIR, "best_model.onnx")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported model to {onnx_path}")
    
    # Generate Metadata
    metadata = {
        "model_name": "lower_limb_ctrgcn",
        "input_shape": [4, 300, 10, 1],
        "output_shape": [len(CLASSES)],
        "normalization_method": "Linear interpolation to 300 frames, coordinate extraction via MediaPipe",
        "class_mapping": {i: c for i, c in enumerate(CLASSES)},
        "joint_mapping": {
            0: "23 Left Hip",
            1: "24 Right Hip",
            2: "25 Left Knee",
            3: "26 Right Knee",
            4: "27 Left Ankle",
            5: "28 Right Ankle",
            6: "29 Left Heel",
            7: "30 Right Heel",
            8: "31 Left Foot Index",
            9: "32 Right Foot Index"
        }
    }
    with open(os.path.join(EXPORT_DIR, "deployment_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    # Generate Final Report
    metrics_path = os.path.join(EVAL_DIR, 'metrics.json')
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
    train_csv = os.path.join(BASE_DIR, "train_labels.csv")
    test_csv = os.path.join(BASE_DIR, "test_labels.csv")
    
    num_train = sum(1 for _ in open(train_csv)) - 1 if os.path.exists(train_csv) else 0
    num_test = sum(1 for _ in open(test_csv)) - 1 if os.path.exists(test_csv) else 0
    
    report_md = f"""# Final Training Report: Lower-Limb CTR-GCN

## Dataset Overview
- **Total Dataset Size**: {num_train + num_test}
- **Train Samples**: {num_train}
- **Test Samples**: {num_test}

## Model Performance
- **Best Accuracy**: {metrics.get('best_accuracy', 0):.4f}
- **Macro F1**: {metrics.get('macro_f1', 0):.4f}
- **Top-3 Accuracy**: {metrics.get('top_3_accuracy', 0):.4f}
- **Hardware Used**: Apple Silicon MPS / CUDA / CPU

## Expected Performance Goals Analysis
Target metrics: Accuracy > 90%, Macro F1 > 0.88, Top-3 > 97%.
"""
    if metrics.get('best_accuracy', 0) < 0.90 or metrics.get('macro_f1', 0) < 0.88:
        report_md += """
### Recommendations for Improvement
The target metrics were not achieved. It is recommended to:
1. **Identify confusing classes**: Check `confusion_matrix.png` and `classification_report.txt` in the `evaluation/` folder.
2. **Additional Data**: Collect more varied videos for the worst-performing classes.
3. **Augmentation**: Apply spatial transformations (e.g., slight joint jittering) or temporal shifting during training.
4. **Class Balancing**: Ensure classes are perfectly balanced via undersampling or weighted loss.
"""
    else:
        report_md += "\n**Status**: Target metrics achieved successfully!\n"
        
    with open("training_report.md", "w") as f:
        f.write(report_md)
        
    print("Phase 8 & 9 Completed. End-to-End Pipeline Finished!")

if __name__ == "__main__":
    export_and_report()
