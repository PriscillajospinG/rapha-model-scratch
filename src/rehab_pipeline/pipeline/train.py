"""
Usage:
    rehab-train --domain lower_limb
    rehab-train --domain upper_body
"""
import os
import csv
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from ..domains import get_domain, DOMAIN_NAMES
from ..common.crypto import load_encrypted_npy

# ----------------------------------------------------
# Dataset
# ----------------------------------------------------
class SkeletonDataset(Dataset):
    def __init__(self, csv_file, skeleton_dir):
        self.skeleton_dir = skeleton_dir
        self.data = []
        with open(csv_file, 'r') as f:
            for row in csv.DictReader(f):
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data[idx]['filename']
        class_id = int(self.data[idx]['class_id'])
        tensor_path = os.path.join(self.skeleton_dir, filename)

        x = load_encrypted_npy(tensor_path).astype(np.float32)
        x = torch.from_numpy(x)
        return x, class_id

# ----------------------------------------------------
# Graph Definition -- generic over any (num_node, edges) skeleton topology
# ----------------------------------------------------
class Graph:
    def __init__(self, num_node, edges):
        self.num_node = num_node
        self.edges = edges
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        A = np.zeros((3, self.num_node, self.num_node))
        for i in range(self.num_node):
            A[0, i, i] = 1
        for i, j in self.edges:
            A[1, i, j] = 1
            A[2, j, i] = 1
        for k in range(3):
            d = np.sum(A[k], axis=1) + 1e-6
            D_inv = np.diag(1.0 / d)
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
    def __init__(self, num_class, in_channels, graph):
        super(CTRGCN, self).__init__()
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
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1).mean(dim=1)
        return self.fc(x)


def build_model(domain):
    graph = Graph(domain.num_joints, domain.graph_edges)
    return CTRGCN(num_class=len(domain.classes), in_channels=4, graph=graph)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available()
                         else "mps" if torch.backends.mps.is_available()
                         else "cpu")


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits_list, labels_list = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        logits_list.append(out.cpu())
        labels_list.append(y)
    return torch.cat(logits_list), torch.cat(labels_list)


def fit_temperature(logits, labels):
    """Post-hoc calibration (Guo et al. 2017): a single scalar T that
    minimizes NLL on held-out (validation) logits. Confidence gating at
    inference time is meaningless if raw softmax is overconfident, which
    small/imbalanced training sets like this one reliably produce."""
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(min=0.05).item())


def confidence_threshold_sweep(probs, labels):
    sweep = []
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    for t in np.arange(0.30, 0.95, 0.05):
        mask = confidences >= t
        coverage = float(mask.mean())
        if mask.sum() == 0:
            acc = None
        else:
            acc = float((preds[mask] == labels[mask]).mean())
        sweep.append({'threshold': round(float(t), 2), 'coverage': coverage, 'accuracy_when_covered': acc})

    recommended = 0.5
    for row in sweep:
        if row['accuracy_when_covered'] is not None and row['accuracy_when_covered'] >= 0.85 and row['coverage'] >= 0.3:
            recommended = row['threshold']
            break
    return sweep, recommended


def train(domain):
    base_dir = os.path.join("datasets", domain.name)
    skeleton_dir = os.path.join(base_dir, "skeletons")
    model_dir = os.path.join("models", domain.name)
    eval_dir = os.path.join("evaluation", domain.name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    device = get_device()
    print(f"[{domain.name}] Using device: {device}")

    train_csv = os.path.join(base_dir, "train_labels.csv")
    val_csv = os.path.join(base_dir, "val_labels.csv")
    test_csv = os.path.join(base_dir, "test_labels.csv")

    if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
        print("Dataset splits not found. Please run phase 5.")
        return

    train_dataset = SkeletonDataset(train_csv, skeleton_dir)
    val_dataset = SkeletonDataset(val_csv, skeleton_dir)
    test_dataset = SkeletonDataset(test_csv, skeleton_dir)

    batch_size = min(64, len(train_dataset) if len(train_dataset) > 0 else 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(domain).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_val_acc = 0.0
    early_stop_counter = 0
    epochs = 200

    train_losses, val_losses, val_accs = [], [], []

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

        # Validation -- used ONLY for early stopping / checkpoint selection.
        # The test set is never touched until training is completely done.
        model.eval()
        val_loss, correct, total = 0, 0, 0
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

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | "
              f"Val Loss: {val_losses[-1]:.4f} | Val Acc: {acc:.4f}")

        torch.save(model.state_dict(), os.path.join(model_dir, "last_model.pth"))

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= 25:
            print("Early stopping triggered!")
            break

    # ------------------------------------------------
    # Post-training: calibration + threshold tuning on VAL, single final
    # evaluation on TEST.
    # ------------------------------------------------
    print("\n--- Generating Evaluation Metrics ---")
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pth"), map_location=device))
    model.eval()

    val_logits, val_labels = collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    print(f"Calibrated temperature (fit on val set): {temperature:.3f}")

    calibrated_val_probs = F.softmax(val_logits / temperature, dim=1).numpy()
    threshold_sweep, recommended_threshold = confidence_threshold_sweep(
        calibrated_val_probs, val_labels.numpy()
    )
    print(f"Recommended confidence threshold (from val sweep): {recommended_threshold}")

    test_logits, test_labels_t = collect_logits(model, test_loader, device)
    test_probs = F.softmax(test_logits / temperature, dim=1).numpy()
    all_preds = test_probs.argmax(axis=1)
    all_labels = test_labels_t.numpy()

    if test_probs.shape[1] >= 3:
        top3 = np.argsort(test_probs, axis=1)[:, -3:]
        top3_acc = float(np.mean([1 if all_labels[i] in top3[i] else 0 for i in range(len(all_labels))]))
    else:
        top3_acc = float((all_preds == all_labels).mean())

    cls_report = classification_report(all_labels, all_preds, target_names=domain.classes,
                                        output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(domain.classes))))

    metrics = {
        'domain': domain.name,
        'best_val_accuracy': best_val_acc,
        'test_accuracy': float((all_preds == all_labels).mean()),
        'macro_f1': cls_report['macro avg']['f1-score'],
        'top_3_accuracy': top3_acc,
        'note': 'best_val_accuracy was used for checkpoint selection; test_accuracy is the '
                'held-out number that was never used for any training decision.',
    }
    with open(os.path.join(eval_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(eval_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(all_labels, all_preds, target_names=domain.classes, zero_division=0))

    with open(os.path.join(eval_dir, 'confusion_matrix.json'), 'w') as f:
        json.dump({'labels': domain.classes, 'matrix': cm.tolist()}, f, indent=4)

    with open(os.path.join(eval_dir, 'calibration.json'), 'w') as f:
        json.dump({
            'temperature': temperature,
            'recommended_confidence_threshold': recommended_threshold,
            'threshold_sweep_on_val': threshold_sweep,
        }, f, indent=4)

    # Confusion matrix plot
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(len(domain.classes)), domain.classes, rotation=45, ha='right')
    plt.yticks(range(len(domain.classes)), domain.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({domain.name}, test set)')
    for i in range(len(domain.classes)):
        for j in range(len(domain.classes)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                      color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, 'confusion_matrix.png'))
    plt.close()

    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title(f'Loss Curve ({domain.name})')
    plt.legend()
    plt.savefig(os.path.join(eval_dir, 'loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(val_accs, label='Val Accuracy')
    plt.title(f'Accuracy Curve ({domain.name})')
    plt.legend()
    plt.savefig(os.path.join(eval_dir, 'accuracy_curve.png'))
    plt.close()

    print(f"Phase 6 & 7 Completed successfully ({domain.name}).")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Phase 6 & 7: Train + evaluate")
    parser.add_argument("--domain", required=True, choices=DOMAIN_NAMES)
    args = parser.parse_args()
    train(get_domain(args.domain))


if __name__ == "__main__":
    main()
