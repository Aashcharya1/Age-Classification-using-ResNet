# Age Classification Project

A deep learning project for classifying face images as "Young" or "Old" using a ResNet-18 architecture trained from scratch.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Implementation Details](#implementation-details)
- [Training Pipeline](#training-pipeline)
- [Hyperparameters](#hyperparameters)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Technical Details](#technical-details)

## Project Overview

This project implements a binary age classifier that predicts whether a person in a face photograph is "Young" (class 0) or "Old" (class 1). The model uses a ResNet-18 backbone trained from scratch (no pretrained weights) on 18,332 training images.

### Key Features
- **ResNet-18 Architecture**: Custom implementation with enhanced classifier head
- **Two-Phase Training**: Initial training with validation monitoring, followed by fine-tuning on combined data
- **Advanced Augmentation**: Comprehensive data augmentation pipeline for better generalization
- **Label Smoothing**: Reduces overconfidence and improves generalization
- **Learning Rate Scheduling**: OneCycleLR for initial training, CosineAnnealingLR for fine-tuning

## Architecture

### Model Structure
The model consists of a ResNet-18 backbone with a custom classifier head:

```python
class MyAgeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # ResNet-18 backbone (from scratch, no pretrained weights)
        self.backbone = models.resnet18(weights=None)
        num_ftrs = self.backbone.fc.in_features  # 512 features
        
        # Custom classifier head
        self.backbone.fc = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),   # Normalizes 512-dim feature vector
            nn.Dropout(0.4),            # 40% dropout for regularization
            nn.Linear(num_ftrs, num_classes)  # 512 → 2 (Young/Old)
        )
```

### Architecture Details
- **Backbone**: ResNet-18 (18 layers, ~11.7M parameters)
- **Feature Extraction**: Outputs 512-dimensional feature vector from average pooling
- **Custom Head**: Replaces default fully connected layer with:
  - **BatchNorm1d**: Normalizes features before classification
  - **Dropout (0.4)**: Regularization to prevent overfitting
  - **Linear Layer**: Maps 512 features → 2 classes
- **Output**: 2 logits (class 0: Young, class 1: Old)

### Forward Pass
1. **Input**: `[batch_size, 3, 224, 224]` RGB images
2. **Processing**: ResNet-18 processes through:
   - Initial convolution layers
   - 4 residual blocks (each with 2 layers)
   - Average pooling
3. **Output**: `[batch_size, 2]` logits

## Dataset

### Dataset Structure
```
dataset/
├── train/
│   ├── 0/          # 9,166 images (Young)
│   └── 1/          # 9,166 images (Old)
├── valid/          # 134 images (flat directory)
└── valid_labels.csv
```

### Dataset Details
- **Training Images**: 18,332 total (9,166 per class)
- **Validation Images**: 134 images
- **Image Format**: 256×256 aligned, cropped face images
- **Labels**:
  - `0` = Young
  - `1` = Old

### Data Loading

#### TrainDataset
Loads images from class subfolders:

```python
class TrainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        for label in [0, 1]:
            cls_dir = os.path.join(root, str(label))
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(IMG_EXT):
                    self.samples.append((os.path.join(cls_dir, fname), label))
```

- Pre-loads all (path, label) pairs into memory
- Sorted filenames for reproducibility
- Supports .png, .jpg, .jpeg formats

#### ValidDataset
Loads images from flat directory with CSV labels:

```python
class ValidDataset(Dataset):
    def __init__(self, root, labels_csv, transform=None):
        # Parse CSV: image,label format
        labels_dict = {}
        with open(labels_csv, 'r') as f:
            for row in csv.DictReader(f):
                labels_dict[row['image']] = int(row['label'])
        
        # Match images with labels
        for fname in sorted(os.listdir(root)):
            if fname in labels_dict:
                self.samples.append((os.path.join(root, fname), labels_dict[fname]))
```

## Implementation Details

### Data Augmentation Pipeline

#### Training Transforms
```python
transforms.Compose([
    # 1. RandomResizedCrop: Random crop + resize
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    #    - Crops random region between 70-100% of image
    #    - Resizes to 224×224
    #    - Introduces scale variation
    
    # 2. RandomHorizontalFlip: Mirror images
    transforms.RandomHorizontalFlip(p=0.5),
    #    - 50% chance to flip horizontally
    #    - Doubles effective dataset size
    
    # 3. ColorJitter: Adjust color properties
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    #    - Brightness: ±20% variation
    #    - Contrast: ±20% variation
    #    - Saturation: ±10% variation
    
    # 4. RandAugment: AutoAugment-style augmentation
    transforms.RandAugment(num_ops=2, magnitude=5),
    #    - Applies 2 random augmentation operations
    #    - Magnitude 5 (moderate strength)
    
    # 5. ToTensor: PIL Image → PyTorch Tensor
    transforms.ToTensor(),
    #    - Converts [0, 255] → [0.0, 1.0]
    #    - Changes HWC → CHW format
    
    # 6. RandomErasing: Cutout-style regularization
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    #    - 20% chance to erase random rectangular region
    #    - Erases 2-10% of image area
    
    # 7. Normalize: ImageNet statistics
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #    - Standardizes pixel values to ~N(0,1)
])
```

**Augmentation Strategy:**
- **Geometric**: RandomResizedCrop, RandomHorizontalFlip
- **Color**: ColorJitter, RandAugment
- **Regularization**: RandomErasing
- **Normalization**: ImageNet statistics

#### Evaluation Transforms
```python
transforms.Compose([
    transforms.Resize((224, 224)),  # Simple resize (no crop)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

- No augmentation during evaluation
- Deterministic resize to 224×224
- Same normalization as training

## Training Pipeline

### Two-Phase Training Strategy

#### Phase 1: Training with Validation
**Objective**: Train model on training set while monitoring validation performance

```python
# Configuration
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1

# Training loop
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, ...)
    val_acc = validate(model, valid_loader, DEVICE)
    
    # Save best model (after epoch 15)
    if epoch >= 15 and val_acc > best_val_acc:
        torch.save(model, 'best_model_phase1.pth')
```

**Details:**
- Trains on 18,332 training images
- Validates on 134 validation images
- Saves best validation checkpoint (after epoch 15)
- Uses OneCycleLR scheduler (per-batch stepping)

#### Phase 2: Fine-tuning on Combined Data
**Objective**: Fine-tune best model on combined training + validation data

```python
# Configuration
FINETUNE_EPOCHS = 30
FINETUNE_LR = 3e-4  # Lower learning rate

# Load best Phase 1 model
model_final = torch.load('best_model_phase1.pth')

# Combine datasets
combined_dataset = ConcatDataset([train_dataset, valid_train_dataset])
# Total: 18,332 + 134 = 18,466 images

# Fine-tune
for epoch in range(1, FINETUNE_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model_final, combined_loader, ...)
    scheduler_final.step()  # CosineAnnealingLR (per-epoch)
```

**Details:**
- Loads best Phase 1 checkpoint
- Combines training + validation (18,466 images)
- Applies training augmentations to validation images
- Fine-tunes for 30 epochs with lower LR (3e-4)
- Uses CosineAnnealingLR scheduler (per-epoch stepping)

### Training Functions

#### Training Function
```python
def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()  # Enable dropout, batch norm training mode
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        # Transfer to GPU (non_blocking=True for async transfer)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Zero gradients (set_to_none=True saves memory)
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        outputs = model(images)  # [batch_size, 2]
        
        # Compute loss
        loss = criterion(outputs, labels)  # CrossEntropyLoss with label smoothing
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Step scheduler per batch (for OneCycleLR)
        if scheduler is not None:
            scheduler.step()

        # Accumulate metrics
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total
```

**Key Features:**
- `non_blocking=True`: Asynchronous CPU→GPU transfer
- `set_to_none=True`: Faster gradient zeroing
- Per-batch scheduler stepping for OneCycleLR
- Metrics accumulated across batches

#### Validation Function
```python
@torch.no_grad()  # Disable gradient computation
def validate(model, loader, device):
    model.eval()  # Disable dropout, use batch norm eval mode
    correct, total = 0, 0
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    
    return correct / total  # Accuracy
```

**Key Features:**
- `@torch.no_grad()`: Disables autograd for efficiency
- `model.eval()`: Sets batch norm and dropout to eval mode
- No loss computation during validation

## Hyperparameters

### Phase 1 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| BATCH_SIZE | 64 | Balance between memory and gradient stability |
| NUM_EPOCHS | 200 | Long training for convergence |
| LEARNING_RATE | 1e-3 | Moderate learning rate (0.001) |
| WEIGHT_DECAY | 1e-4 | L2 regularization (0.0001) |
| LABEL_SMOOTH | 0.1 | Label smoothing factor |
| IMG_SIZE | 224 | Input image size |

**Rationale:**
- **Batch Size 64**: Good GPU utilization, stable gradients
- **200 Epochs**: Sufficient for convergence from scratch
- **LR 1e-3**: Works well with OneCycleLR scheduler
- **Weight Decay 1e-4**: Moderate L2 regularization
- **Label Smoothing 0.1**: Reduces overconfidence

### Phase 2 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| FINETUNE_EPOCHS | 30 | Shorter fine-tuning period |
| FINETUNE_LR | 3e-4 | Lower LR (0.0003) for gentle updates |

**Rationale:**
- **Lower LR**: Fine-tuning with smaller updates
- **30 Epochs**: Sufficient for convergence on combined data

### Optimizer Configuration
```python
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LEARNING_RATE,        # 1e-3
    weight_decay=WEIGHT_DECAY  # 1e-4
)
```

- **Adam Optimizer**: Adaptive learning rates per parameter
- **Weight Decay**: L2 regularization

### Learning Rate Schedulers

#### Phase 1: OneCycleLR
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,              # Peak LR: 1e-3
    steps_per_epoch=len(train_loader), # Steps per epoch
    epochs=NUM_EPOCHS                  # Total epochs: 200
)
```

**OneCycleLR Behavior:**
- **Warmup**: LR increases from 0 to max_lr
- **Decay**: LR decreases to near 0
- Stepped **per batch** (not per epoch)
- Helps escape local minima

#### Phase 2: CosineAnnealingLR
```python
scheduler_final = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_final, 
    T_max=FINETUNE_EPOCHS  # Cosine decay over 30 epochs
)
```

**CosineAnnealingLR Behavior:**
- **Smooth decay**: LR follows cosine curve from initial to 0
- Stepped **per epoch** (not per batch)
- Gentle fine-tuning schedule

## Usage

### Prerequisites
```bash
pip install torch torchvision pillow numpy
```

### Training
Run the training script:

```bash
python train.py
```

The script will:
1. Train the model for 200 epochs (Phase 1)
2. Save the best validation model
3. Fine-tune on combined data for 30 epochs (Phase 2)
4. Save final submission files:
   - `{ROLL_NO}.pth` - Trained model
   - `{ROLL_NO}.py` - Model class definition

### Evaluation
Evaluate your model using the provided script:

```bash
python evaluate_submission_student.py \
    --model_path roll_no.pth \
    --model_file roll_no.py \
    --data_dir dataset/
```

### Configuration
Edit `train.py` to modify:
- `ROLL_NO`: Your roll number for submission files
- Hyperparameters (batch size, learning rate, etc.)
- Number of epochs
- Data paths

## File Structure

```
Age Classifier/
├── dataset/
│   ├── train/
│   │   ├── 0/          # Young images
│   │   └── 1/          # Old images
│   ├── valid/          # Validation images
│   └── valid_labels.csv
├── model_class.py      # Model architecture definition
├── train.py           # Main training script
├── evaluate_submission_student.py  # Evaluation script
├── starter_notebook.ipynb  # Starter notebook (reference)
├── b23es1001.ipynb    # Training demonstration notebook
├── b23es1001.pth      # Trained model weights
├── b23es1001.py       # Model class (submission)
├── assignment.txt     # Assignment requirements
└── README.md          # This file
```

### Key Files
- **model_class.py**: Defines `MyAgeClassifier` class with ResNet-18 backbone
- **train.py**: Main training script with two-phase training (includes data augmentation transforms)
- **evaluate_submission_student.py**: Script to evaluate model accuracy
- **b23es1001.ipynb**: Complete training demonstration notebook with 10-epoch demo

## Technical Details

### Reproducibility
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

- Fixed seed (42) for reproducibility
- `cudnn.benchmark = True`: Optimizes for consistent input sizes
- `deterministic = False`: Allows non-deterministic ops for speed

### DataLoader Configuration
```python
_kw = dict(
    pin_memory=torch.cuda.is_available(),  # Pin memory for faster GPU transfer
    num_workers=NUM_WORKERS                 # 0 on Windows, 4 on Linux
)

if NUM_WORKERS > 0:
    _kw['persistent_workers'] = True  # Keep workers alive between epochs

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,           # Shuffle training data
    drop_last=True,         # Drop incomplete last batch
    **_kw
)
```

**Details:**
- `pin_memory=True`: Faster CPU→GPU transfer
- `num_workers`: 0 on Windows (multiprocessing issues), 4 on Linux
- `persistent_workers=True`: Avoids worker recreation overhead
- `drop_last=True`: Consistent batch sizes
- `shuffle=True`: Only for training data

### Memory & Performance Optimizations
- `non_blocking=True`: Asynchronous GPU transfers
- `set_to_none=True`: Faster gradient zeroing
- `@torch.no_grad()`: Disables autograd during validation
- `drop_last=True`: Consistent batch sizes
- `pin_memory=True`: Faster data transfer
- `persistent_workers=True`: Avoids worker overhead

### Model Saving

#### Saving Best Model (Phase 1)
```python
torch.save(model, 'best_model_phase1.pth')
```

- Saves full model (not state_dict)
- Includes architecture and weights
- Required for submission format

#### Final Submission Files
```python
# Save final model
save_path = f'{ROLL_NO}.pth'
torch.save(model_final, save_path)

# Copy model class definition
shutil.copy('model_class.py', f'{ROLL_NO}.py')
```

- `{ROLL_NO}.pth`: Full trained model
- `{ROLL_NO}.py`: Model class definition (required for loading)

#### Sanity Check
```python
test_model = torch.load(save_path, map_location='cpu', weights_only=False)
test_model.eval()
dummy = torch.randn(1, 3, 224, 224)
out = test_model(dummy)
print(f'Output shape: {out.shape}  (expected [1, 2])')
```

- Verifies model loads correctly
- Tests forward pass
- Confirms output shape `[1, 2]`

## Design Decisions & Rationale

1. **ResNet-18 from Scratch**: Assignment requirement (no pretrained weights)
2. **Custom Head with BatchNorm + Dropout**: Regularization to prevent overfitting
3. **Label Smoothing (0.1)**: Reduces overconfidence and improves generalization
4. **Two-Phase Training**:
   - Phase 1: Monitor validation performance
   - Phase 2: Fine-tune on all available data
5. **OneCycleLR (Phase 1)**: Helps escape local minima with warmup and decay
6. **CosineAnnealingLR (Phase 2)**: Smooth fine-tuning with gradual decay
7. **Aggressive Augmentation**: Improves generalization to unseen data
8. **Dropout 0.4**: Balances regularization vs. model capacity
9. **Batch Size 64**: Good GPU utilization without memory issues
10. **200 Epochs**: Sufficient for convergence from scratch

## Training Flow Summary

```
1. Initialize:
   - Set random seeds
   - Load datasets (TrainDataset, ValidDataset)
   - Create DataLoaders
   - Initialize model, optimizer, scheduler

2. Phase 1 (200 epochs):
   - Train on 18,332 images
   - Validate on 134 images
   - Save best model (after epoch 15)
   - OneCycleLR scheduler (per-batch)

3. Phase 2 (30 epochs):
   - Load best Phase 1 model
   - Combine train + validation (18,466 images)
   - Fine-tune with lower LR (3e-4)
   - CosineAnnealingLR scheduler (per-epoch)

4. Save:
   - Final model: {ROLL_NO}.pth
   - Model class: {ROLL_NO}.py
   - Sanity check: verify model loads correctly
```

## Results

The model achieves **93.28% accuracy** on the validation set through:
- Careful hyperparameter tuning
- Comprehensive data augmentation
- Two-phase training strategy
- Regularization techniques (dropout, label smoothing, weight decay)

## License

This project is part of a Deep Learning course assignment (Spring 2026).

## Author

**Roll Number**: b23es1001

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [torchvision Models](https://pytorch.org/vision/stable/models.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Label Smoothing](https://arxiv.org/abs/1512.00567)
- [OneCycleLR](https://arxiv.org/abs/1708.07120)

