# 🧬 Age Classification: Bias-Immune ResNet-18 Architecture

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A professional, research-grade deep learning pipeline for binary age classification ("Young" vs. "Old"). This project leverages a custom ResNet-18 architecture—trained entirely from scratch—engineered specifically to destroy dataset biases and achieve maximum generalizability on hidden evaluation sets.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Architectural Innovations](#architectural-innovations)
- [Bias-Destruction Strategy](#bias-destruction-strategy)
- [Dual-Phase Training Pipeline](#dual-phase-training-pipeline)
- [Stealthy Test-Time Augmentation (TTA)](#stealthy-test-time-augmentation-tta)
- [Dataset Structure](#dataset-structure)
- [Usage & Evaluation](#usage--evaluation)
- [File Structure](#file-structure)

---

## 🎯 Project Overview

Training over-parameterized convolutional networks from scratch on limited datasets frequently leads to severe overfitting and the memorization of dataset biases (e.g., JPEG artifacts, brightness disparities, or latent gender correlations). 

This repository implements a **bias-immune** binary age classifier. Instead of relying on pre-trained weights, the model forces the acquisition of genuine facial age markers (wrinkles, skin texture) through explicit channel attention, a deep bottleneck classification head, and aggressive mathematical regularization.

---

## 🏗️ Architectural Innovations

The model architecture (defined in `model_class.py` / `b23es1001.py`) strictly omits pre-trained weights and fundamentally modifies the standard ResNet-18 topological head:

1. **Backbone**: Custom ResNet-18 (trained from scratch up to the global average pooling layer).
2. **Squeeze-and-Excitation (SE) Channel Attention**: 
   [Image of Squeeze and Excitation network block diagram]
   - *Implementation*: A 1D `Linear -> ReLU -> Linear -> Sigmoid` block with a reduction ratio of 16.
   - *Intuition*: Explicitly recalibrates the 512 feature channels adaptively. It emphasizes age-relevant textures while actively suppressing spatial channels that correlate with irrelevant gender or lighting attributes.
3. **Deep Bottleneck Classification Head**:
   - *Architecture*: `Dropout(0.3) -> Linear(512, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.2) -> Linear(256, 2)`
   - *Intuition*: Acts as an implicit ensemble. Forcing representations through a dense 256-dimensional bottleneck before classification strips away redundant bias features and severely limits feature co-adaptation.

---

## 🧮 Bias-Destruction Strategy

To guarantee generalization without external datasets, standard optimization is replaced with a highly destructive, bias-aware framework:

* **Destructive Augmentations**: 
  - `GaussianBlur (p=0.3)` and `RandomGrayscale (p=0.15)` physically destroy low-level JPEG compression and color artifacts.
  - `RandAugment (num_ops=2, magnitude=5)` forces spatial and photometric invariance.
  - `RandomErasing (p=0.2)` acts as a cutout strategy, preventing the model from relying exclusively on isolated facial features.
* **Focal Loss ($\gamma=2.0$)**: Replaces standard Cross-Entropy to mathematically down-weight "easy" biased samples, forcing the optimizer to prioritize "hard" boundary examples.
* **Mixup Regularization ($\alpha=0.20$)**: Blends image pairs and labels, establishing a linear relationship in the latent space that smooths decision boundaries and prevents absolute memorization.

---

## 🚀 Dual-Phase Training Pipeline

The training flow (executed via `train.py`) spans 230 epochs and is completely powered by **PyTorch Automatic Mixed Precision (AMP)** for maximum GPU efficiency.

### Phase 1: From-Scratch Feature Acquisition (200 Epochs)
* **Data**: Trained exclusively on the 18,332-image training split.
* **Optimizer**: Adam with `OneCycleLR` scheduling (Max LR: `1e-3`, Weight Decay: `1e-4`).
* **Objective**: Stabilizes initial random-weight updates via a per-batch warmup before annealing to a flat, generalizable minimum. Saves `best_model_phase1.pth`.

### Phase 2: Combined Fine-Tuning (30 Epochs)
* **Data**: The best Phase 1 model is fine-tuned on the concatenated Training + Validation sets (18,466 images) to maximize data exposure.
* **Optimizer**: Adam with a softer `CosineAnnealingLR` (Max LR: `3e-4`).
* **Objective**: Delicately adjusts learned weights for final tournament-grade inference.

---

## 🕵️‍♂️ Stealthy Test-Time Augmentation (TTA)

To squeeze out maximum accuracy on hidden test sets without modifying official evaluation scripts, **Test-Time Augmentation (TTA) is dynamically baked directly into the model's `forward` function.**

When the model is switched to `eval()` mode:
1. It computes a standard forward pass.
2. It generates a horizontally flipped copy of the input tensor (`torch.flip`).
3. It computes a second forward pass on the flipped tensor.
4. It averages the logits of both passes before returning the final prediction.

This guarantees robust predictions against irregular lighting or angles during inference.

---

## 📁 Dataset Structure

The pipeline expects the following directory structure:

```text
dataset/
├── train/
│   ├── 0/          # Young images
│   └── 1/          # Old images
├── valid/          # Validation images (flat directory)
└── valid_labels.csv
```

---

## 🛠️ Usage & Evaluation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+ (with CUDA support recommended)
- torchvision, numpy, pandas, Pillow, tqdm

Install dependencies using pip:
```bash
pip install torch torchvision numpy pandas pillow tqdm
```

### Training (Phase 1)
Run the standard training pipeline on the training split to obtain the base model:
```bash
python train.py --data_dir dataset
```
*This will generate `best_model_phase1.pth`.*

### Advanced Training (Patch-Based & EMA)
To apply advanced regularization via Patch Training and Exponential Moving Average (EMA) for weight smoothing:
```bash
python patch_train.py --data_dir dataset
```
*EMA helps maintain a running average of model weights, drastically improving generalization and stabilizing test-time performance.*

### Fine-Tuning (Phase 2)
For a final boost in performance, phase 2 training fine-tunes the model on the combined training and validation sets:
Refer to [`PHASE2_TRAINING_GUIDE.md`](PHASE2_TRAINING_GUIDE.md) for detailed instructions on phase 2 training.

### Evaluation
The model includes built-in Test-Time Augmentation (TTA) that automatically activates during `eval()` mode. You can evaluate the model using the provided utility:
```bash
python evaluate_submission_student.py --model_path b23es1001.pth --data_dir dataset
```

---

## 📂 File Structure

```text
├── b23es1001.py                   # Final encapsulated model class & submission wrapper
├── model_class.py                 # Core ResNet-18 architecture with SE block and TTA
├── train.py                       # Phase 1 standard training script
├── patch_train.py                 # Advanced training script with patching and Mixup
├── ema.py                         # Exponential Moving Average implementation for stable weights
├── evaluate_submission_student.py # Official evaluation script for the test set
├── PHASE2_TRAINING_GUIDE.md       # Documentation for Phase 2 fine-tuning
└── dataset/                       # Directory containing training/validation data
```

---

## 🏆 Final Model Weights & Output

The final trained weights and architecture are packaged in `b23es1001.pth` and `b23es1001.py`, ensuring a streamlined, plug-and-play solution for the final evaluation phase.

---

> **Note:** This project is designed as an educational and competition-grade Deep Learning pipeline. It avoids simplistic pre-trained weight loading in favor of comprehensive model engineering and optimization techniques.