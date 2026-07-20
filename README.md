# Skeleton-Based Action Recognition Training Pipeline

This repository contains the complete pipeline for extracting skeletons from physiotherapy videos and training a CTR-GCN model on an NVIDIA GPU.

## Environment Setup (NVIDIA GPU)
To train this model on a cloud instance with an NVIDIA GPU (e.g., L4, T4, A100), ensure you have NVIDIA drivers and CUDA installed, then set up the environment:

```bash
# 1. Create and activate a python environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with CUDA support (Important for NVIDIA GPUs)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt
```

## Running the Training Pipeline

### 1. Data Preparation
Make sure your dataset structure looks like this:
```
datasets/lower_limb/skeletons/  <-- Contains all extracted .npy skeleton files
datasets/lower_limb/            <-- Contains train_labels.csv and test_labels.csv
```
If you haven't split the dataset yet, run:
```bash
python phase_5_split.py
```

### 2. Train the Model
The training script will automatically detect and utilize the NVIDIA GPU (`cuda`). We use a deep CTR-GCN architecture with early stopping enabled.

```bash
python phase_6_7_train.py
```
* **Output**: The best PyTorch weights will be saved to `models/lower_limb/best_model.pth`.

### 3. Export for Deployment
Once training is complete, export the model to ONNX format. ONNX is highly optimized and can be run in production without PyTorch.

```bash
python phase_8_9_export.py
```
* **Outputs**: 
  * `models/lower_limb/best_model.onnx`
  * `models/lower_limb/deployment_metadata.json`
  * `evaluation/lower_limb/` (Metrics and Graphs)

## Notes on Model Capacity
This pipeline uses a highly robust Channel-wise Topology Refinement Graph Convolutional Network (CTR-GCN). Because of the depth of the model, you must feed it a sufficient amount of data to achieve >90% accuracy. We recommend at least **150-200 videos per class**.
