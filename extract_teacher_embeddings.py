"""
Extract teacher embeddings offline using CLIP or DINOv2.
This script processes all training images and saves their embeddings.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# Try to import CLIP, fallback to DINOv2 if not available
try:
    import clip
    USE_CLIP = True
except ImportError:
    USE_CLIP = False
    try:
        import torchvision
        USE_DINOV2 = True
    except ImportError:
        USE_DINOV2 = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224


class TrainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        for label in [0, 1]:
            cls_dir = os.path.join(root, str(label))
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith(".png"):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path


def get_teacher_model():
    """Load a pretrained teacher model for feature extraction."""
    if USE_CLIP:
        print("Using CLIP as teacher model...")
        model, preprocess = clip.load("ViT-B/32", device=DEVICE)
        model.eval()
        return model, preprocess, 512  # CLIP ViT-B/32 has 512-dim embeddings
    elif USE_DINOV2:
        print("Using DINOv2 as teacher model...")
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        model.to(DEVICE)
        model.eval()
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        return model, preprocess, 768  # DINOv2 ViT-B/14 has 768-dim embeddings
    else:
        raise ImportError("Neither CLIP nor DINOv2 available. Install with: pip install git+https://github.com/openai/CLIP.git")


@torch.no_grad()
def extract_embeddings(model, dataloader, embed_dim, use_clip=False):
    """Extract embeddings for all images."""
    embeddings = {}
    labels = {}
    
    for images, labels_batch, paths in tqdm(dataloader, desc="Extracting embeddings"):
        images = images.to(DEVICE)
        
        if use_clip:
            # CLIP returns image features directly
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)  # Normalize
        else:
            # DINOv2 forward pass
            features = model(images)
            features = features / features.norm(dim=-1, keepdim=True)  # Normalize
        
        # Store embeddings by image path
        for i, path in enumerate(paths):
            img_name = os.path.basename(path)
            embeddings[img_name] = features[i].cpu()
            labels[img_name] = labels_batch[i].item()
    
    return embeddings, labels


def main():
    DATA_ROOT = "dataset/"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    OUTPUT_DIR = "teacher_embeddings"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load teacher model
    teacher_model, preprocess, embed_dim = get_teacher_model()
    
    # Create dataset and dataloader
    dataset = TrainDataset(TRAIN_DIR, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    print(f"Extracting embeddings for {len(dataset)} images...")
    print(f"Embedding dimension: {embed_dim}")
    
    # Extract embeddings
    embeddings, labels = extract_embeddings(
        teacher_model, dataloader, embed_dim, use_clip=USE_CLIP
    )
    
    # Save embeddings
    embeddings_path = os.path.join(OUTPUT_DIR, "train_embeddings.pt")
    labels_path = os.path.join(OUTPUT_DIR, "train_labels.pt")
    
    torch.save(embeddings, embeddings_path)
    torch.save(labels, labels_path)
    
    print(f"\nSaved embeddings to {embeddings_path}")
    print(f"Saved labels to {labels_path}")
    print(f"Total embeddings: {len(embeddings)}")


if __name__ == "__main__":
    main()
