import os
import csv
import json
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
# Dataset
# ----------------------------------------------------
class SkeletonDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
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
        
        # Load (4, 300, 10, 1)
        x = np.load(tensor_path).astype(np.float32)
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
        
        return self.fc(x)

# ----------------------------------------------------
# Training Loop
# ----------------------------------------------------
def train():
    setup_dirs()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_csv = os.path.join(BASE_DIR, "train_labels.csv")
    test_csv = os.path.join(BASE_DIR, "test_labels.csv")
    
    if not os.path.exists(train_csv):
        print("Dataset splits not found. Please run phase 5.")
        return
        
    train_dataset = SkeletonDataset(train_csv)
    test_dataset = SkeletonDataset(test_csv)
    
    # Since we are running on an L4 GPU, we can comfortably use a larger batch size for faster training
    batch_size = min(64, len(train_dataset) if len(train_dataset) > 0 else 1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = CTRGCN(num_class=len(CLASSES)).to(device)
    
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
        
        # Eval
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
        acc = correct / max(total, 1)
        val_losses.append(val_loss / len(test_loader))
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
    
    # Save Metrics JSON
    metrics = {
        'best_accuracy': best_acc,
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
