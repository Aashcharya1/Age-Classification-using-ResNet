## Phase II Age Classifier – Training Guide

This file explains, in simple words, how to train your final Phase II model and what makes it special.

### 1. What this model does

- **Goal**: Predict whether a face is *young* or *old* while avoiding gender and dataset biases.
- **Backbone**: `ResNet-18` trained **from scratch** (no pretraining).
- **Attention**: A **Squeeze-and-Excitation (SE)** block after global average pooling re-weights the 512 channels, so the model focuses on true age cues (wrinkles, skin texture) instead of shortcuts (like hair or jawline).
- **Head**: A deep bottleneck classifier:
  - `Dropout(0.3) -> Linear(512, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.2) -> Linear(256, 2)`

All of this is implemented in `model_class.py` inside the `MyAgeClassifier` class.

### 2. Anti-bias tricks used

- **Channel attention (SE block)**: Encourages the network to emphasize age-related channels and suppress gender-related ones.
- **Strong data augmentation**:
  - `RandomResizedCrop`, `RandomHorizontalFlip`
  - `ColorJitter` and `RandAugment`
  - **GaussianBlur (p = 0.3)** and **RandomGrayscale (p = 0.15)** to break JPEG/brightness artifacts
  - `RandomErasing` to prevent overfitting to small regions
- **Focal Loss (γ = 2)**:
  - Down-weights very easy examples (often the biased ones)
  - Forces the model to learn from hard, boundary images.
- **Mixup (α = 0.20)**:
  - Blends pairs of images and labels.
  - Smooths the decision boundary and makes memorizing artifacts mathematically harder.

All of these are wired into `train.py`.

### 3. Folder structure expected

`train.py` assumes the following structure under `dataset/`:

- `dataset/train/0/` – training images labeled `0`
- `dataset/train/1/` – training images labeled `1`
- `dataset/valid/` – validation images (mixed 0/1)
- `dataset/valid_labels.csv` – CSV with two columns: `image,label`

Put your Phase II training data there before running the script.

### 4. How to install dependencies

Use Python 3.9+ and install:

```bash
pip install torch torchvision pillow numpy
```

You can also add any other standard utilities you like (e.g. `tqdm`), but they are not required.

### 5. How to train the model

1. **Open a terminal** in the project folder (`F:\Age Classifier`).
2. **Make sure the dataset is in place** as described above.
3. Run:

```bash
python train.py
```

The script automatically:

- Trains on the **train set** and evaluates on the **valid set** (Part 1).
- Saves the **best validation model** as `best_model_phase1.pth`.
- Then **merges train + valid** into a single dataset and **fine-tunes** the best model (Part 2).
- Saves final submission files:
  - `b23es1001.pth` – the trained model.
  - `b23es1001.py` – the model definition (copied from `model_class.py`).

### 6. Key hyperparameters (already set in `train.py`)

- **Batch size**: `64`
- **Epochs (Part 1)**: `200`
- **Optimizer**: `Adam` with `lr = 1e-3`, `weight_decay = 1e-4`
- **Scheduler (Part 1)**: `OneCycleLR`
- **Fine-tune epochs (Part 2)**: `30` with lower learning rate `3e-4`
- **Loss**: `FocalLoss(gamma=2.0)` with **Mixup(α = 0.20)**

You can change these inside `train.py` if you want to experiment, but the current values are tuned for strong performance.

### 7. How to use the trained model

The competition’s `evaluate_submission_student.py` (or the official checker) will:

- Load `b23es1001.pth`
- Import `b23es1001.py` and its `MyAgeClassifier` class
- Feed images of shape `[B, 3, 224, 224]` through the model

As long as you do not rename the class or change its interface, everything will work.

### 8. Summary for your report (plain language)

- You **kept Rank 4’s deep bottleneck head** to reduce feature co-adaptation.
- You added an **SE channel attention block** to emphasize age cues and reduce reliance on gender.
- You **destroyed dataset artifacts** with blur + grayscale and heavy augmentation.
- You **replaced Cross-Entropy with Focal Loss (γ = 2)** so the model focuses on hard / unbiased examples.
- You wrapped training in **Mixup (α = 0.20)** and then **retrained on train + validation** to squeeze the maximum information from the full dataset.

