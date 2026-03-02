# Age Classification Assignment - Implementation Details

## Overview

This project implements a binary age classifier (Young vs Old) using a ResNet-18 backbone trained from scratch. The implementation incorporates multiple advanced deep learning techniques to maximize classification accuracy on the given face image dataset.

---

## What Has Been Done

### 1. **Model Architecture**

**Base Architecture:** ResNet-18 (trained from scratch, no pretrained weights)

**Modifications:**
- **Heavy Classification Head**: Instead of a simple linear layer, a more sophisticated head is used:
  - `Linear(512 → 256)` projection layer
  - `BatchNorm1d(256)` for normalization
  - `GELU` activation function
  - `Dropout(0.5)` for regularization
  - Final `Linear(256 → 2)` classifier

**Test Time Augmentation (TTA):**
- Built-in horizontal flip augmentation during inference
- Predictions are averaged: `(original + flipped) / 2`
- Automatically applied when model is in `eval()` mode

### 2. **Data Augmentation**

**Training Augmentation Pipeline:**
- `RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1))` - Random cropping with resizing
- `RandomHorizontalFlip(p=0.5)` - Horizontal flipping
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05)` - Color variations (30% probability)
- `RandomGrayscale(p=0.05)` - Random grayscale conversion
- `RandAugment(num_ops=2, magnitude=9)` - Automated augmentation policy
- `RandomErasing(p=0.15, scale=(0.02, 0.15))` - Random erasing for robustness
- Standard ImageNet normalization: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

**Validation Augmentation:**
- Simple resize to 224×224
- Normalization only (no random augmentations)

### 3. **Advanced Training Techniques**

#### 3.1 MixUp and CutMix Regularization
- **MixUp**: Linearly interpolates between two images and their labels
  - Alpha parameter: 0.4
  - Ensures lambda ≥ 0.5 (original image dominates)
- **CutMix**: Replaces a random rectangular patch with another image's patch
  - Alpha parameter: 1.0
  - Lambda adjusted based on actual patch area
- Applied with 50% probability per batch (randomly chooses MixUp or CutMix)

#### 3.2 Loss Functions

**Label Smoothing Cross-Entropy:**
- Smoothing factor: 0.1
- Prevents overconfidence and improves generalization
- Softens hard labels: `(1 - smoothing) * one_hot + smoothing / (num_classes - 1)`

**Optional Teacher Guidance (Combined Loss):**
- Combines label-smoothed CE with MSE on normalized embeddings
- Alpha: 0.6 (weight on CE loss)
- Uses teacher embeddings from foundation models (if available)
- Includes projection layer if teacher and student feature dimensions differ

#### 3.3 Optimizer and Learning Rate Scheduling

**Adam Optimizer:**
- Learning rate: 3e-3 (maximum for OneCycleLR)
- Weight decay: 0.05 (L2 regularization)

**OneCycleLR Scheduler:**
- Cosine annealing strategy
- Warm-up phase: 10% of total steps
- Gradually increases LR to max, then decreases
- Total steps: `num_epochs × steps_per_epoch`

#### 3.4 Exponential Moving Average (EMA)
- Decay rate: 0.999
- Maintains a smoothed version of model weights
- Shadow weights updated after each optimizer step
- Applied during validation and final model saving
- Improves model stability and generalization

#### 3.5 Mixed Precision Training (AMP)
- Automatic Mixed Precision using `torch.cuda.amp`
- Faster training with reduced memory usage
- Gradient scaling to prevent underflow
- Enabled automatically when CUDA is available

#### 3.6 Gradient Clipping
- Maximum gradient norm: 1.0
- Prevents exploding gradients
- Applied after gradient unscaling in AMP

#### 3.7 BatchNorm Calibration
- After applying EMA weights, BatchNorm running statistics are recalibrated
- Runs 80 forward passes through training data in train mode
- Ensures BN statistics match the EMA weights
- Critical for proper inference with EMA models

### 4. **Training Strategy**

**Phase 1: Train on Training Set**
- Train for 100 epochs on training data only
- Validate using EMA weights + TTA
- Save best model based on validation accuracy

**Phase 2: Retrain on Combined Data**
- Combine training + validation data
- Train fresh model for 100 epochs on combined dataset
- Apply EMA + BN calibration before saving
- This is the final submission model

### 5. **Reproducibility**
- Fixed random seed: 42
- Seeds set for: Python random, NumPy, PyTorch (CPU and CUDA)
- CUDNN benchmark enabled for speed (non-deterministic)

---

## Concepts Used

### 1. **Transfer Learning Concepts**
- While pretrained weights are not used, the ResNet-18 architecture benefits from proven design patterns
- Heavy head design allows the backbone to learn rich features while the head specializes

### 2. **Regularization Techniques**
- **Data Augmentation**: Increases dataset diversity
- **MixUp/CutMix**: Creates synthetic training examples
- **Dropout**: Prevents co-adaptation of neurons
- **Label Smoothing**: Reduces overfitting to hard labels
- **Weight Decay**: L2 regularization on parameters

### 3. **Optimization Techniques**
- **OneCycleLR**: Super-convergence learning rate schedule
- **Adam**: Standard Adam optimizer with L2 regularization
- **Gradient Clipping**: Stabilizes training
- **EMA**: Smooths weight updates for better generalization

### 4. **Inference Techniques**
- **Test Time Augmentation**: Improves prediction robustness
- **BatchNorm Calibration**: Ensures proper statistics with EMA weights

### 5. **Memory and Speed Optimization**
- **Mixed Precision Training**: Reduces memory and speeds up training
- **Efficient Data Loading**: Pin memory, persistent workers (on non-Windows)

---

## Key Files

1. **`starter_notebook.ipynb`**: Main training notebook
   - Dataset loading
   - Model initialization
   - Training loops (Phase 1 and Phase 2)
   - Model saving

2. **`model_class.py`**: Model architecture definition
   - `AgeClassifier` class with ResNet-18 backbone
   - Heavy head with TTA
   - Feature extraction methods

3. **`training_utils.py`**: Utility functions
   - Data augmentation pipelines
   - MixUp/CutMix implementations
   - Loss functions (Label Smoothing, Combined Loss)
   - EMA class
   - BatchNorm calibration

4. **`evaluate_submission_student.py`**: Evaluation script
   - Loads saved model
   - Runs inference on validation/test set
   - Computes accuracy

---

## Hyperparameters Summary

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 224×224 | Input image resolution |
| Batch Size | 64 | Training batch size |
| Epochs | 100 | Number of training epochs |
| Learning Rate | 3e-3 | Maximum LR for OneCycleLR |
| Weight Decay | 0.05 | Adam weight decay |
| MixUp Alpha | 0.4 | MixUp interpolation parameter |
| CutMix Alpha | 1.0 | CutMix interpolation parameter |
| Mix Probability | 0.5 | Probability of applying MixUp/CutMix |
| Label Smoothing | 0.1 | Label smoothing factor |
| EMA Decay | 0.999 | Exponential moving average decay |
| Gradient Clip | 1.0 | Maximum gradient norm |
| Dropout | 0.5 | Dropout probability in head |

---

## References

### Papers

1. **ResNet**: He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
   - Original ResNet architecture

2. **MixUp**: Zhang, H., et al. "mixup: Beyond Empirical Risk Minimization." ICLR 2018.
   - Data augmentation by mixing images and labels

3. **CutMix**: Yun, S., et al. "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features." ICCV 2019.
   - Patch-based augmentation strategy

4. **Label Smoothing**: Szegedy, C., et al. "Rethinking the Inception Architecture for Computer Vision." CVPR 2016.
   - Regularization technique for classification

5. **OneCycleLR**: Smith, L. N. "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates." 2017.
   - Learning rate scheduling strategy

6. **Adam**: Kingma, D. P., & Ba, J. "Adam: A Method for Stochastic Optimization." ICLR 2015.
   - Adaptive learning rate optimization algorithm

7. **RandAugment**: Cubuk, E. D., et al. "RandAugment: Practical Automated Data Augmentation with a Reduced Search Space." NeurIPS 2020.
   - Automated augmentation policy

8. **Random Erasing**: Zhong, Z., et al. "Random Erasing Data Augmentation." AAAI 2020.
   - Regularization through random erasing

9. **EMA**: Polyak, B. T., & Juditsky, A. B. "Acceleration of Stochastic Approximation by Averaging." SIAM Journal on Control and Optimization, 1992.
   - Exponential moving average for model weights

10. **GELU**: Hendrycks, D., & Gimpel, K. "Gaussian Error Linear Units (GELUs)." 2016.
    - Activation function used in the head

### Libraries and Tools

- **PyTorch**: Deep learning framework
  - Documentation: https://pytorch.org/docs
- **Torchvision**: Computer vision utilities
  - Models: https://pytorch.org/vision/stable/models
- **PIL/Pillow**: Image processing
- **NumPy**: Numerical computations

### Online Resources

- PyTorch Official Documentation: https://pytorch.org/docs
- Torchvision Models: https://pytorch.org/vision/stable/models
- PyTorch Tutorials: https://pytorch.org/tutorials

---

## Design Decisions

### Why Heavy Head?
- Allows the ResNet-18 backbone to learn general features
- The head can specialize for the binary classification task
- Dropout and BatchNorm in head provide additional regularization

### Why TTA?
- Horizontal flipping is natural for face images (faces are roughly symmetric)
- Averaging predictions reduces variance and improves robustness
- Minimal computational overhead

### Why EMA?
- Smooths out noisy weight updates during training
- Often leads to better generalization
- Common practice in state-of-the-art models

### Why MixUp/CutMix?
- Creates more diverse training examples
- Helps model learn more robust features
- Reduces overfitting

### Why OneCycleLR?
- Faster convergence (super-convergence)
- Better final performance
- Automatically handles warm-up and annealing

### Why Label Smoothing?
- Prevents overconfidence
- Improves generalization
- Particularly useful for binary classification

---

## Results and Observations

The implementation follows best practices for deep learning training:

1. **Strong Augmentation**: Multiple augmentation techniques increase data diversity
2. **Regularization**: Multiple regularization techniques prevent overfitting
3. **Optimization**: Modern optimizer and scheduler for efficient training
4. **Inference**: TTA and EMA improve final predictions
5. **Two-Phase Training**: Phase 1 for validation, Phase 2 for final submission

The model is designed to maximize accuracy on the hidden test set while following all assignment constraints (ResNet-18 backbone, trained from scratch, no pretrained weights).

---

## Notes

- The model uses ResNet-18 as required by the assignment
- All weights are trained from scratch (no pretrained models)
- The implementation is fully compliant with assignment requirements
- Teacher guidance is optional and can be enabled if teacher embeddings are available
- The code is designed to work on both Windows and Linux systems
