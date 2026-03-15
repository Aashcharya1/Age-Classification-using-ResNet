"""
Age Classification EMA Fine-Tuning Script (Phase II)
====================================================
This script skips Phase 1 and loads 'best_model_phase1.pth'.
It fine-tunes the model on the combined dataset for 30 epochs
while maintaining an Exponential Moving Average (EMA) of the weights.
"""
import os
import csv
import random
import shutil
import platform
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch.amp as amp
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torch.nn.functional as F

from model_class import MyAgeClassifier

# ──────────────────── Reproducibility ────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ──────────────────── Configuration ────────────────────
DATA_ROOT    = 'dataset/'
TRAIN_DIR    = os.path.join(DATA_ROOT, 'train')
VALID_DIR    = os.path.join(DATA_ROOT, 'valid')
VALID_LABELS = os.path.join(DATA_ROOT, 'valid_labels.csv')
IMG_SIZE     = 224
IMG_EXT      = ('.png', '.jpg', '.jpeg')

# Phase 2 Hyperparameters
BATCH_SIZE    = 64
FINETUNE_EPOCHS = 30
FINETUNE_LR   = 3e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 0 if platform.system() == 'Windows' else 4
ROLL_NO       = 'b23es1001'

# ──────────────────── Datasets ────────────────────
class TrainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        for label in [0, 1]:
            cls_dir = os.path.join(root, str(label))
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(IMG_EXT):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class ValidDataset(Dataset):
    def __init__(self, root, labels_csv, transform=None):
        self.transform = transform
        self.samples = []
        labels_dict = {}
        with open(labels_csv, 'r') as f:
            for row in csv.DictReader(f):
                labels_dict[row['image']] = int(row['label'])
        for fname in sorted(os.listdir(root)):
            if fname.lower().endswith(IMG_EXT) and fname in labels_dict:
                self.samples.append((os.path.join(root, fname), labels_dict[fname]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ──────────────────── Transforms ────────────────────
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandAugment(num_ops=2, magnitude=5),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
    transforms.RandomGrayscale(p=0.15),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ──────────────────── Loss & Mixup ────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def mixup_data(x, y, alpha=0.20):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ──────────────────── Training Logic ────────────────────
def train_one_epoch_ema(model, ema_model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup augmentation
        images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.20)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == 'cuda':
            with amp.autocast('cuda'):
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()

        # ---> UPDATE EMA EVERY BATCH <---
        ema_model.update_parameters(model)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct_a = (preds == labels_a).sum().item()
        correct_b = (preds == labels_b).sum().item()
        correct += (lam * correct_a + (1 - lam) * correct_b)
        total += images.size(0)

    return total_loss / total, correct / total

# ──────────────────── Main Execution ────────────────────
if __name__ == '__main__':
    print('\n' + '='*60)
    print('  Part 2: EMA Fine-Tuning on Combined Data (Train + Valid)')
    print('='*60 + '\n')

    # 1. Combine Datasets
    train_dataset = TrainDataset(TRAIN_DIR, transform=train_transform)
    valid_train_dataset = ValidDataset(VALID_DIR, VALID_LABELS, transform=train_transform)
    combined_dataset = ConcatDataset([train_dataset, valid_train_dataset])
    
    _kw = dict(pin_memory=torch.cuda.is_available(), num_workers=NUM_WORKERS)
    if NUM_WORKERS > 0:
        _kw['persistent_workers'] = True
        
    combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **_kw)
    print(f'Combined Dataset: {len(combined_dataset)} images')

    # 2. Load Checkpoint
    checkpoint_path = 'best_model_phase1.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Cannot find {checkpoint_path}! Make sure you ran Phase 1 first.")
    
    print(f'Loading {checkpoint_path}...')
    model_final = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_final.to(DEVICE)

    # 3. Initialize EMA Model
    ema_model = AveragedModel(model_final, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    
    # 4. Setup Optimizer & Scheduler
    criterion_final = FocalLoss(gamma=2.0) 
    optimizer_final = optim.Adam(model_final.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler_final = optim.lr_scheduler.CosineAnnealingLR(optimizer_final, T_max=FINETUNE_EPOCHS)
    scaler_final = amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    print(f'\nStarting EMA Fine-tuning for {FINETUNE_EPOCHS} epochs...')
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch_ema(
            model_final, ema_model, combined_loader, optimizer_final, criterion_final, DEVICE, scaler_final
        )
        scheduler_final.step()

        lr = optimizer_final.param_groups[0]['lr']
        if epoch % 5 == 0 or epoch <= 3 or epoch == FINETUNE_EPOCHS:
            print(f'Epoch {epoch:03d}/{FINETUNE_EPOCHS}  Loss {train_loss:.4f}  Train {train_acc*100:.1f}%  LR {lr:.6f}')

    print('\n' + '='*60)
    print('  Saving EMA Submission Files')
    print('='*60 + '\n')

    # 5. Save the EMA Weights (Not the raw model)
    save_path = f'{ROLL_NO}.pth'
    torch.save(ema_model.module, save_path)
    mb = os.path.getsize(save_path) / 1e6
    print(f'Saved smoothed EMA model to {save_path} ({mb:.1f} MB)')
    
    shutil.copy('model_class.py', f'{ROLL_NO}.py')
    print(f'Copied model_class.py -> {ROLL_NO}.py')
    print('\n✓ EMA Fine-tuning complete! You can now run evaluate_submission_student.py')