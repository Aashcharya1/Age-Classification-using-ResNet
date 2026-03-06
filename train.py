"""
Age Classification Training Script
===================================
ResNet-18 from scratch | Adam | OneCycleLR | Label Smoothing

Part 1: Train on train set, checkpoint best val model with fine tuning from pretrained model

Usage:
    python train.py
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

# Hyperparameters
BATCH_SIZE    = 64
NUM_EPOCHS    = 200            
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4           
LABEL_SMOOTH  = 0.1           

# Workers (0 on Windows)
NUM_WORKERS = 0 if platform.system() == 'Windows' else 4

# Submission
ROLL_NO = 'b23es1001'

print(f'\nEpochs: {NUM_EPOCHS}  |  LR: {LEARNING_RATE}  |  WD: {WEIGHT_DECAY}')
print(f'Batch: {BATCH_SIZE}  |  Label smoothing: {LABEL_SMOOTH}')
print(f'Workers: {NUM_WORKERS}')


# ──────────────────── Datasets ────────────────────

class TrainDataset(Dataset):
    """Load training images from class sub-folders (0/ and 1/)."""
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
    """Load validation images from flat directory + CSV labels."""
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
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),  # Scale variation
    transforms.RandomHorizontalFlip(p=0.5),                     # Horizontal mirroring
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # Color variations
    transforms.RandAugment(num_ops=2, magnitude=5),              # AutoAugment-style
    transforms.ToTensor(),                                       # PIL to Tensor
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),         # Cutout regularization
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet stats
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),                     # Simple resize
    transforms.ToTensor(),                                       # PIL to Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet stats
])

print(f'\nTrain transform: {train_transform}')
print(f'Eval transform:  {eval_transform}')


# ──────────────────── Training & Validation Functions ────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """Standard training loop for one epoch with per-batch scheduler stepping."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Step scheduler per batch (for OneCycleLR)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, device):
    """Evaluate accuracy on validation set."""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def validate_with_tta(model, loader, device, num_views=5):
    """Evaluate accuracy with Test-Time Augmentation (TTA)."""
    model.eval()
    correct, total = 0, 0
    
    # Use functional transforms for tensor-based augmentation
    from torchvision.transforms.functional import hflip
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        batch_size = images.size(0)
        # Store predictions for original + augmented views
        all_preds = torch.zeros((batch_size, 2), device=device)
        
        # 1. Original image pass
        all_preds += torch.softmax(model(images), dim=1)
        
        # 2. Augmented passes - horizontal flips
        for view_idx in range(num_views - 1):
            aug_images = images.clone()
            
            # Horizontal flip (deterministic pattern for consistency)
            if view_idx % 2 == 0:
                # Flip all images
                aug_images = hflip(aug_images)
            # For odd indices, use original (already added)
            
            all_preds += torch.softmax(model(aug_images), dim=1)
        
        # Average predictions and get final answer
        all_preds /= num_views
        predictions = all_preds.argmax(dim=1)
        
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    return correct / total

 
#  Part 1: Train on training set, evaluate on validation set 

print('\n' + '='*60)
print('  Part 1: Training with validation')
print('='*60 + '\n')

# Datasets & loaders
train_dataset = TrainDataset(TRAIN_DIR, transform=train_transform)
valid_dataset = ValidDataset(VALID_DIR, VALID_LABELS, transform=eval_transform)

_kw = dict(pin_memory=torch.cuda.is_available(), num_workers=NUM_WORKERS)
if NUM_WORKERS > 0:
    _kw['persistent_workers'] = True

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=True, **_kw)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **_kw)

print(f'Train: {len(train_dataset)} images  ({len(train_loader)} batches)')
print(f'Valid: {len(valid_dataset)} images  ({len(valid_loader)} batches)\n')

# Model, loss, optimizer, scheduler
model = MyAgeClassifier(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
 
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS
)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Parameters: {n_params:,}')
print(f'Optimizer:  Adam (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})')
print(f'Scheduler:  OneCycleLR (max_lr={LEARNING_RATE}, steps_per_epoch={len(train_loader)})')
print(f'Loss:       CrossEntropyLoss (label_smoothing={LABEL_SMOOTH})\n')

# Training loop with best-model checkpointing
best_val_acc = 0.0
best_epoch = 0
MIN_EPOCH_FOR_BEST = 15  # Don't save anything until epoch 15

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scheduler)
    val_acc = validate(model, valid_loader, DEVICE)

    saved = '' 
    if epoch >= MIN_EPOCH_FOR_BEST and val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model, 'best_model_phase1.pth')
        saved = f'  ** BEST {val_acc*100:.2f}%'

    lr = optimizer.param_groups[0]['lr']
    if epoch % 10 == 0 or epoch <= 5 or saved:
        print(f'Epoch {epoch:03d}/{NUM_EPOCHS}  '
              f'Loss {train_loss:.4f}  Train {train_acc*100:.1f}%  '
              f'Val {val_acc*100:.2f}%  LR {lr:.6f}{saved}')

print(f'\nPart 1 done — best val accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}')
 

#  Part 2: Fine-tune best model on train + valid combined 

print('\n' + '='*60)
print('  Part 2: Fine-tune on combined data (train + valid)')
print('='*60 + '\n')

FINETUNE_EPOCHS = 30
FINETUNE_LR = 3e-4  # Lower learning rate for gentle fine-tuning

# Combine train + valid (apply training augmentations to valid images!)
valid_train_dataset = ValidDataset(VALID_DIR, VALID_LABELS, transform=train_transform)
combined_dataset = ConcatDataset([train_dataset, valid_train_dataset])

combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=True, **_kw)

print(f'Combined: {len(combined_dataset)} images  ({len(combined_loader)} batches)')
print(f'Fine-tuning for {FINETUNE_EPOCHS} epochs with LR={FINETUNE_LR}\n')

# Load the best Part 1 model to fine-tune
model_final = torch.load('best_model_phase1.pth', map_location=DEVICE, weights_only=False)
model_final.to(DEVICE)

criterion_final = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH) 
optimizer_final = optim.Adam(model_final.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
# CosineAnnealing steps per epoch, not per batch
scheduler_final = optim.lr_scheduler.CosineAnnealingLR(optimizer_final, T_max=FINETUNE_EPOCHS)

for epoch in range(1, FINETUNE_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(
        model_final, combined_loader, optimizer_final, criterion_final, DEVICE,
        scheduler=None  
    )
    scheduler_final.step()  # Step per epoch

    lr = optimizer_final.param_groups[0]['lr']
    if epoch % 5 == 0 or epoch <= 3 or epoch == FINETUNE_EPOCHS:
        print(f'Epoch {epoch:03d}/{FINETUNE_EPOCHS}  '
              f'Loss {train_loss:.4f}  Train {train_acc*100:.1f}%  LR {lr:.6f}')

print('\Part 2 fine-tuning complete.')

 
#  Save submission files 

print('\n' + '='*60)
print('  Saving submission files')
print('='*60 + '\n')

# Save final model (full model, NOT state_dict)
save_path = f'{ROLL_NO}.pth'
torch.save(model_final, save_path)
mb = os.path.getsize(save_path) / 1e6
print(f'Saved model to {save_path} ({mb:.1f} MB)')
 
shutil.copy('model_class.py', f'{ROLL_NO}.py')
print(f'Copied model_class.py -> {ROLL_NO}.py')

print(f'\nSubmission files ready:')
print(f'  - {ROLL_NO}.pth  (trained model)')
print(f'  - {ROLL_NO}.py   (model class definition)')
print(f'  - {ROLL_NO}.pdf  (create this manually — 1 page report)')

# Quick sanity check: load the model back
print(f'\nSanity check: loading {save_path}...')
test_model = torch.load(save_path, map_location='cpu', weights_only=False)
test_model.eval()
dummy = torch.randn(1, 3, 224, 224)
out = test_model(dummy)
print(f'Output shape: {out.shape}  (expected [1, 2])')
print('✓ Model loads and runs correctly!')
