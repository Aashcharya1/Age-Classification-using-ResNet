"""
Training utilities: augmentation, MixUp, CutMix, losses, EMA.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

try:
    from torchvision.transforms import v2
    RandAugment = v2.RandAugment
except (ImportError, AttributeError):
    from torchvision.transforms import RandAugment


# ──────────────────── Augmentation ────────────────────

def get_advanced_augmentation(img_size=224):
    """
    Strong training augmentation pipeline:
    RandomResizedCrop + Flip + ColorJitter + Grayscale + RandAugment + Erase
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05
            )
        ], p=0.3),
        transforms.RandomGrayscale(p=0.05),
        RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])


def get_eval_transform(img_size=224):
    """Standard evaluation transform (matches evaluation script)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ──────────────────── MixUp / CutMix ────────────────────

def mixup_data(x, y, alpha=0.3):
    """
    MixUp augmentation. Ensures lam >= 0.5 so original dominates.

    Returns: mixed_x, y_a (original), y_b (shuffled), lam, shuffle_index
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  # keep lam >= 0.5
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam, idx


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix augmentation: paste a random patch from shuffled images.

    Returns: mixed_x, y_a (original), y_b (shuffled), lam (adjusted), shuffle_index
    """
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    r = np.sqrt(1 - lam)
    rh, rw = int(H * r), int(W * r)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1 = max(0, cy - rh // 2)
    y2 = min(H, cy + rh // 2)
    x1 = max(0, cx - rw // 2)
    x2 = min(W, cx + rw // 2)
    out = x.clone()
    out[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)  # actual mix ratio
    return out, y, y[idx], lam, idx


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixed (MixUp/CutMix) samples."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ──────────────────── Loss Functions ────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing for better generalization."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=1)
        n = pred.size(1)
        with torch.no_grad():
            dist = torch.zeros_like(log_prob).fill_(self.smoothing / (n - 1))
            dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-dist * log_prob, dim=1))


class CombinedLoss(nn.Module):
    """
    Label-smoothed CE + optional teacher guidance (MSE on normalized embeddings).

    When called without features/names (e.g., for mixed batches),
    returns plain CE — no alpha scaling.
    """

    def __init__(self, alpha=0.6, label_smoothing=0.1,
                 teacher_embeddings=None, student_feat_dim=256):
        super().__init__()
        self.alpha = alpha
        self.ce = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.teacher_embeddings = teacher_embeddings
        self.projection = None

        if teacher_embeddings and len(teacher_embeddings) > 0:
            first = next(iter(teacher_embeddings.values()))
            t_dim = first.shape[0] if isinstance(first, torch.Tensor) else len(first)
            if t_dim != student_feat_dim:
                self.projection = nn.Linear(t_dim, student_feat_dim)
                print(f"  Projection layer: {t_dim} -> {student_feat_dim}")

    def forward(self, pred, target, features=None, names=None):
        ce_loss = self.ce(pred, target)

        # No teacher guidance → return plain CE (no alpha scaling)
        if features is None or names is None or not self.teacher_embeddings:
            return ce_loss, ce_loss, torch.tensor(0.0, device=pred.device)

        # Collect matching teacher embeddings
        t_embs, s_idx = [], []
        for i, n in enumerate(names):
            if n in self.teacher_embeddings:
                e = self.teacher_embeddings[n]
                t_embs.append(e if isinstance(e, torch.Tensor) else
                              torch.tensor(e, dtype=torch.float32))
                s_idx.append(i)

        if not t_embs:
            return ce_loss, ce_loss, torch.tensor(0.0, device=pred.device)

        t_batch = torch.stack(t_embs).to(features.device)
        if self.projection is not None:
            t_batch = self.projection(t_batch)

        t_loss = F.mse_loss(
            F.normalize(features[s_idx], dim=1),
            F.normalize(t_batch, dim=1)
        )
        total = self.alpha * ce_loss + (1 - self.alpha) * t_loss
        return total, ce_loss, t_loss


# ──────────────────── EMA ────────────────────

class EMA:
    """
    Exponential Moving Average of model parameters.
    Maintains a shadow copy that is a smoothed version of the weights.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """Update shadow weights after each optimizer step."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def apply_shadow(self, model):
        """Replace model params with EMA shadow weights (for eval/save)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original model params after eval."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ──────────────────── BN Calibration ────────────────────

@torch.no_grad()
def calibrate_bn(model, loader, device, steps=50):
    """
    Recalibrate BatchNorm running stats after applying EMA weights.
    Run a few forward passes through training data in train mode.
    """
    # Reset BN running stats
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()

    model.train()
    for i, (imgs, *_rest) in enumerate(loader):
        if i >= steps:
            break
        model(imgs.to(device))
    model.eval()
