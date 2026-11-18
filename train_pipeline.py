import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from dataset import NuclearCataractDatabase
from models.googlenet import CataractGoogLeNet

# Maximize CPU parallelism
num_cpus = os.cpu_count() or 8
torch.set_num_threads(num_cpus)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ImageNet normalization transform
norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class FastCachedDataset(Dataset):
    """Ultra-compact uint8 numpy cached dataset with 158MB total RAM footprint"""
    def __init__(self, full_ds, indices, augment=False):
        self.augment = augment
        n_samples = len(indices)
        self.images = np.zeros((n_samples, 224, 224, 3), dtype=np.uint8)
        self.labels = np.zeros(n_samples, dtype=np.int64)
        self.filenames = []

        self.aug_pipeline = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ])

        print(f"Pre-loading {n_samples} samples into compact uint8 RAM buffer...", flush=True)
        for idx_count, i in enumerate(indices):
            item = full_ds[i]
            self.images[idx_count] = item['orig_img']
            self.labels[idx_count] = item['label']
            self.filenames.append(item['filename'])
        print("Pre-loading complete. Total buffer size: 158MB.", flush=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_np = self.images[idx]
        label = self.labels[idx]
        
        # Convert uint8 numpy array to float32 tensor [0, 1]
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        if self.augment:
            tensor = self.aug_pipeline(tensor)

        norm_tensor = norm_transform(tensor)

        return {
            'tensor': norm_tensor,
            'label': label,
            'filename': self.filenames[idx]
        }

def train_model(epochs_per_fold=10, n_splits=5):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} (Threads: {torch.get_num_threads()})", flush=True)

    # Load base dataset
    full_ds = NuclearCataractDatabase()
    n_samples = len(full_ds)
    print(f"Total dataset size: {n_samples} images", flush=True)

    # Extract patient IDs for patient-level grouping
    patient_ids = [sample.get('patient_id', 'unknown') for sample in full_ds.samples]
    unique_patients = list(set(patient_ids))
    print(f"Total unique patients: {len(unique_patients)}", flush=True)

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_splits)

    all_targets = [sample['label'] for sample in full_ds.samples]
    X_dummy = np.zeros(n_samples)

    os.makedirs('checkpoints', exist_ok=True)
    best_checkpoint_path = 'checkpoints/googlenet_cataract.pth'

    overall_best_acc = 0.0
    fold_metrics = []

    print(f"\nStarting Patient-Level {n_splits}-Fold Cross-Validation CataractGoogLeNet Training...", flush=True)
    print("=" * 70, flush=True)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_dummy, all_targets, groups=patient_ids), 1):
        print(f"\n--- FOLD {fold}/{n_splits} ---", flush=True)
        print(f"Training images: {len(train_idx)} | Validation images: {len(val_idx)}", flush=True)
        train_patients = set(patient_ids[i] for i in train_idx)
        val_patients = set(patient_ids[i] for i in val_idx)
        print(f"Training patients: {len(train_patients)} | Validation patients: {len(val_patients)}", flush=True)

        train_ds = FastCachedDataset(full_ds, train_idx, augment=True)
        val_ds = FastCachedDataset(full_ds, val_idx, augment=False)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

        model = CataractGoogLeNet(num_classes=4, pretrained=True).to(device)
        for param in model.parameters():
            param.requires_grad = True

        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
        head_params = [p for n, p in model.named_parameters() if 'fc' in n]

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1.5e-4},
            {'params': head_params, 'lr': 8e-4}
        ], weight_decay=1e-4)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_fold, eta_min=1e-5)

        fold_best_acc = 0.0

        for epoch in range(1, epochs_per_fold + 1):
            model.train()
            running_loss = 0.0
            train_preds, train_targets = [], []

            for batch in train_loader:
                inputs = batch['tensor'].to(device)
                targets = torch.as_tensor(batch['label'], dtype=torch.long).to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                preds = torch.argmax(outputs, dim=1)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(targets.cpu().numpy())

            scheduler.step()

            train_epoch_loss = running_loss / len(train_ds)
            train_epoch_acc = accuracy_score(train_targets, train_preds) * 100.0

            # Validation Phase
            model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []

            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch['tensor'].to(device)
                    targets = torch.as_tensor(batch['label'], dtype=torch.long).to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    preds = torch.argmax(outputs, dim=1)
                    val_loss += loss.item() * inputs.size(0)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())

            val_epoch_loss = val_loss / len(val_ds)
            val_epoch_acc = accuracy_score(val_targets, val_preds) * 100.0
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_targets, val_preds, average='weighted', zero_division=0
            )

            print(f"Fold {fold} Epoch [{epoch:02d}/{epochs_per_fold:02d}] "
                  f"Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.2f}% | "
                  f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}% | "
                  f"F1: {f1:.4f}", flush=True)

            if val_epoch_acc > fold_best_acc:
                fold_best_acc = val_epoch_acc
                if fold_best_acc > overall_best_acc:
                    overall_best_acc = fold_best_acc
                    torch.save(model.state_dict(), best_checkpoint_path)
                    print(f"  >>> New Overall Best Checkpoint Saved! Val Acc: {overall_best_acc:.2f}%", flush=True)

        # Audit for this fold
        fold_metrics.append({
            'fold': fold,
            'train_images': len(train_idx),
            'val_images': len(val_idx),
            'val_acc': fold_best_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cm': confusion_matrix(val_targets, val_preds)
        })

    # Overall Audit Report across folds
    avg_acc = np.mean([m['val_acc'] for m in fold_metrics])
    avg_prec = np.mean([m['precision'] for m in fold_metrics])
    avg_rec = np.mean([m['recall'] for m in fold_metrics])
    avg_f1 = np.mean([m['f1'] for m in fold_metrics])
    total_cm = sum([m['cm'] for m in fold_metrics])
    last_val_count = fold_metrics[-1]['val_images']
    last_train_count = fold_metrics[-1]['train_images']

    print("\n" + "=" * 70, flush=True)
    print("=== FINAL AUDIT REPORT (Patient-Level Group 5-Fold Cross Validation) ===", flush=True)
    print(f"Number of Training Images (per fold avg) : {last_train_count}", flush=True)
    print(f"Number of Validation Images (per fold avg): {last_val_count}", flush=True)
    print(f"Total Unique Patients                     : {len(unique_patients)}", flush=True)
    print(f"5-Fold Mean Validation Accuracy           : {avg_acc:.2f}%", flush=True)
    print(f"Peak Validation Accuracy                  : {overall_best_acc:.2f}%", flush=True)
    print(f"Weighted Precision                       : {avg_prec:.4f}", flush=True)
    print(f"Weighted Recall                          : {avg_rec:.4f}", flush=True)
    print(f"Weighted F1-Score                        : {avg_f1:.4f}", flush=True)
    print("Aggregated Confusion Matrix Across Folds:\n", total_cm, flush=True)
    print("=" * 70, flush=True)

    return avg_acc

if __name__ == '__main__':
    train_model(epochs_per_fold=6, n_splits=5)
