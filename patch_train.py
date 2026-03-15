import re
import torch

with open('train.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update transforms
content = re.sub(
    r'train_transform = transforms.Compose\(\[.*?\n\]\)',
    '''train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandAugment(num_ops=2, magnitude=5),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
    transforms.RandomGrayscale(p=0.15),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])''',
    content, flags=re.DOTALL
)

# 2. Add FocalLoss and Mixup definitions
focal_loss_mixup_def = '''
import torch.nn.functional as F

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

'''

content = re.sub(
    r'(# ─* Training & Validation Functions ─*\s*)',
    focal_loss_mixup_def + r'\1',
    content
)

# 3. Modify train_one_epoch
new_train_epoch = '''def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup augmentation
        images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.20)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        
        # Approximate accuracy during mixup
        preds = outputs.argmax(1)
        correct_a = (preds == labels_a).sum().item()
        correct_b = (preds == labels_b).sum().item()
        correct += (lam * correct_a + (1 - lam) * correct_b)
        total += images.size(0)

    return total_loss / total, correct / total
'''

content = re.sub(
    r'def train_one_epoch\(model, loader, optimizer, criterion, device, scheduler=None\):.*?return total_loss / total, correct / total',
    new_train_epoch,
    content, flags=re.DOTALL
)

# 4. Replace CrossEntropyLoss with FocalLoss in both parts
content = re.sub(
    r'criterion = nn\.CrossEntropyLoss\(label_smoothing=LABEL_SMOOTH\)',
    'criterion = FocalLoss(gamma=2.0)',
    content
)
content = re.sub(
    r'criterion_final = nn\.CrossEntropyLoss\(label_smoothing=LABEL_SMOOTH\)',
    'criterion_final = FocalLoss(gamma=2.0)',
    content
)

with open('train_modified.py', 'w', encoding='utf-8') as f:
    f.write(content)
