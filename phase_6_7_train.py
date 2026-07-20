import os
import csv
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

BASE_DIR = "datasets/lower_limb"
SKELETON_DIR = os.path.join(BASE_DIR, "skeletons")
MODEL_DIR = "models/lower_limb"
EVAL_DIR = "evaluation/lower_limb"

CLASSES = [
    "ankle", "calf", "hamstring", "heel_slide",
    "hip", "knee", "leg_raise", "quadriceps", "toes"
]

def setup_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

# ----------------------------------------------------
# Skeleton preprocessing: root-centering, scale normalization,
# velocity features, and (train-only) online augmentation.
# ----------------------------------------------------
HIP_JOINTS = (0, 1)      # left_hip, right_hip
ANKLE_JOINTS = (4, 5)     # left_ankle, right_ankle


def normalize_position(pos):
    """
    pos: np.ndarray (3, T, V, M) raw [x, y, z]
    Returns root-centered + scale-normalized pos, same shape. Centering on
    the per-frame mid-hip and scaling by mean hip-to-ankle "body length"
    removes camera distance/framing as a nuisance variable.
    """
    pos = pos.copy()
    hip_center = pos[:, :, HIP_JOINTS, :].mean(axis=2, keepdims=True)  # (3, T, 1, 1)
    pos = pos - hip_center

    left_d = np.linalg.norm(pos[:, :, HIP_JOINTS[0], :] - pos[:, :, ANKLE_JOINTS[0], :], axis=0)
    right_d = np.linalg.norm(pos[:, :, HIP_JOINTS[1], :] - pos[:, :, ANKLE_JOINTS[1], :], axis=0)
    scale = float(np.concatenate([left_d, right_d]).mean())
    scale = max(scale, 1e-6)
    return pos / scale


def add_velocity(x):
    """
    x: np.ndarray (4, T, V, M) = normalized [x, y, z, visibility]
    Returns (7, T, V, M): x concatenated with 3 velocity channels [vx, vy, vz]
    computed from the (already-normalized/augmented) position channels.
    """
    pos = x[:3]
    vel = np.zeros_like(pos)
    vel[:, 1:, :, :] = pos[:, 1:, :, :] - pos[:, :-1, :, :]
    return np.concatenate([x, vel], axis=0).astype(np.float32)


def augment_skeleton(pos):
    """
    Applied to the *normalized* position channels (3, T, V, M) only -- train
    split, called fresh every epoch. Must run AFTER normalize_position, not
    before: normalization rescales by a hip-to-ankle distance measured from
    the same tensor, so a scale perturbation applied before normalization
    would just be measured and divided back out. Small enough to preserve
    the exercise's identity while adding real training diversity, unlike the
    near-no-op video-level h-flip/brightness augmentation used to pad the
    raw dataset.
    """
    C, T, V, M = pos.shape

    # Small in-plane rotation (camera-angle variance).
    angle = math.radians(np.random.uniform(-10, 10))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    x, y = pos[0].copy(), pos[1].copy()
    pos[0] = x * cos_a - y * sin_a
    pos[1] = x * sin_a + y * cos_a

    # Small uniform scale jitter.
    pos *= np.random.uniform(0.9, 1.1)

    # Gaussian joint jitter.
    pos += np.random.normal(0, 0.02, size=pos.shape).astype(np.float32)

    # Mild temporal crop + edge-pad (speed/duration perturbation).
    min_len = int(T * 0.9)
    crop_len = np.random.randint(min_len, T + 1)
    if crop_len < T:
        start = np.random.randint(0, T - crop_len + 1)
        cropped = pos[:, start:start + crop_len]
        pad = np.repeat(cropped[:, -1:, :, :], T - crop_len, axis=1)
        pos = np.concatenate([cropped, pad], axis=1)

    return pos


# ----------------------------------------------------
# Dataset
# ----------------------------------------------------
class SkeletonDataset(Dataset):
    def __init__(self, csv_file, augment=False):
        self.data = []
        self.augment = augment
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data[idx]['filename']
        class_id = int(self.data[idx]['class_id'])
        tensor_path = os.path.join(SKELETON_DIR, filename)

        # Load (4, 300, 10, 1) = [x, y, z, visibility]
        raw = np.load(tensor_path).astype(np.float32)
        pos = normalize_position(raw[:3])
        vis = raw[3:4]

        if self.augment:
            pos = augment_skeleton(pos)

        x = add_velocity(np.concatenate([pos, vis], axis=0))
        x = torch.from_numpy(x)
        return x, class_id

# ----------------------------------------------------
# Graph Definition for 10 Joints
# ----------------------------------------------------
class Graph:
    def __init__(self):
        self.num_node = 10
        self.edges = [
            (0, 1), (0, 2), (1, 3), (2, 4), (3, 5),
            (4, 6), (5, 7), (4, 8), (5, 9), (6, 8), (7, 9)
        ]
        self.A = self.get_adjacency_matrix()
        
    def get_adjacency_matrix(self):
        A = np.zeros((3, self.num_node, self.num_node))
        # 0: identity, 1: inward, 2: outward
        for i in range(self.num_node):
            A[0, i, i] = 1
            
        for i, j in self.edges:
            A[1, i, j] = 1 # Inward
            A[2, j, i] = 1 # Outward
            
        # Normalize
        for k in range(3):
            d = np.sum(A[k], axis=1) + 1e-6
            d_inv = 1.0 / d
            D_inv = np.diag(d_inv)
            A[k] = np.dot(D_inv, A[k])
            
        return torch.tensor(A, dtype=torch.float32)

# ----------------------------------------------------
# CTR-GCN Model (Simplified variant)
# ----------------------------------------------------
class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(CTRGC, self).__init__()
        self.num_subset = A.shape[0]
        self.out_channels = out_channels
        
        self.conv = nn.Conv2d(in_channels, out_channels * self.num_subset, kernel_size=1)
        self.A = nn.Parameter(A.clone())
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        N, C, T, V = x.size()
        x = self.conv(x)
        x = x.view(N, self.num_subset, self.out_channels, T, V)
        
        y = torch.einsum('n k c t v, k v w -> n c t w', x, self.A)
        y = self.bn(y)
        y = self.relu(y)
        return y

class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(TCN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CTRGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1):
        super(CTRGCNBlock, self).__init__()
        self.gcn = CTRGC(in_channels, out_channels, A)
        self.tcn = TCN(out_channels, out_channels, stride=stride)
        
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = lambda x: x
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.relu(x + res)

class CTRGCN(nn.Module):
    def __init__(self, num_class=9, in_channels=4, graph=None):
        super(CTRGCN, self).__init__()
        if graph is None:
            graph = Graph()
        A = graph.A
        
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        
        self.l1 = CTRGCNBlock(in_channels, 64, A, stride=1)
        self.l2 = CTRGCNBlock(64, 64, A, stride=1)
        self.l3 = CTRGCNBlock(64, 128, A, stride=2)
        self.l4 = CTRGCNBlock(128, 128, A, stride=1)
        self.l5 = CTRGCNBlock(128, 256, A, stride=2)

        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)

        # Global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)

        x = self.drop(x)
        return self.fc(x)

# ----------------------------------------------------
# Training Loop
# ----------------------------------------------------
def train():
    setup_dirs()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_csv = os.path.join(BASE_DIR, "train_labels.csv")
    val_csv = os.path.join(BASE_DIR, "val_labels.csv")
    test_csv = os.path.join(BASE_DIR, "test_labels.csv")

    if not os.path.exists(train_csv) or not os.path.exists(val_csv) or not os.path.exists(test_csv):
        print("Dataset splits not found. Please run phase 5.")
        return

    # augment=True only on the train split -- val/test must see the same
    # deterministic, un-augmented inputs every time so model selection and
    # the final report are both honest.
    train_dataset = SkeletonDataset(train_csv, augment=True)
    val_dataset = SkeletonDataset(val_csv, augment=False)
    test_dataset = SkeletonDataset(test_csv, augment=False)

    # Since we are running on an L4 GPU, we can comfortably use a larger batch size for faster training
    batch_size = min(64, len(train_dataset) if len(train_dataset) > 0 else 1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CTRGCN(num_class=len(CLASSES), in_channels=7).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    best_acc = 0.0
    early_stop_counter = 0
    epochs = 200
    
    train_losses = []
    val_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        train_losses.append(total_loss / len(train_loader))
        scheduler.step()

        # Eval on the held-out validation split (never the test split --
        # test stays untouched until the final report below).
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
        acc = correct / max(total, 1)
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {acc:.4f}")
        
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "last_model.pth"))
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= 25:
            print("Early stopping triggered!")
            break
            
    # Evaluation phase
    print("\n--- Generating Evaluation Metrics ---")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth")))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            
    # Top-3 Accuracy
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    if all_probs.shape[1] >= 3:
        top3 = np.argsort(all_probs, axis=1)[:, -3:]
        top3_acc = np.mean([1 if all_labels[i] in top3[i] else 0 for i in range(len(all_labels))])
    else:
        top3_acc = accuracy_score(all_labels, all_preds) # Fallback
        
    cls_report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True, zero_division=0)
    test_acc = accuracy_score(all_labels, all_preds)

    # Save Metrics JSON. 'best_accuracy' is the TEST-set accuracy of the
    # checkpoint that was selected using validation accuracy (best_acc) --
    # test data is never used for checkpoint selection or early stopping.
    metrics = {
        'best_accuracy': test_acc,
        'best_val_accuracy': best_acc,
        'macro_f1': cls_report['macro avg']['f1-score'],
        'top_3_accuracy': top3_acc
    }
    with open(os.path.join(EVAL_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    with open(os.path.join(EVAL_DIR, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0))
        
    # Plot Loss Curve
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(EVAL_DIR, 'loss_curve.png'))
    plt.close()
    
    # Plot Accuracy Curve
    plt.figure()
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(EVAL_DIR, 'accuracy_curve.png'))
    plt.close()
    
    print("Phase 6 & 7 Completed successfully.")
    
if __name__ == "__main__":
    train()
