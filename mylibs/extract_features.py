"""
Phase 1: Extract DINO features from the Flowers dataset and cache to disk.
Run once: python extract_features.py
Outputs: features.npy, labels.npy, image_paths.npy, val_indices.npy
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BASE_DIR   = Path(__file__).parent   # mylibs/
IMAGE_DIR  = BASE_DIR / 'data' / 'flowers'
CLASSES    = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
BATCH_SIZE = 64
VAL_FRAC   = 0.15
SEED       = 42


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class FlowersRawDataset(Dataset):
    """Loads raw flower images with their class labels."""

    def __init__(self, image_dir, classes, transform=None):
        self.transform = transform
        self.samples   = []   # list of (path, label_int)

        for label, cls in enumerate(classes):
            folder = Path(image_dir) / cls
            for path in sorted(folder.glob('*.jpg')):
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, str(path)


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def extract_dino_features(batch_size: int = BATCH_SIZE):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # DINO ViT-S/16 — outputs (B, 384)
    print("Loading DINO ViT-S/16 from torch hub …")
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = FlowersRawDataset(IMAGE_DIR, CLASSES, transform=transform)
    if len(dataset) == 0:
        raise FileNotFoundError(
            f"No images found under {IMAGE_DIR.resolve()}. "
            f"Expected subfolders {CLASSES} each containing *.jpg files."
        )
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=(device.type == 'cuda'))

    print(f"Extracting features for {len(dataset)} images …")
    features_list, labels_list, paths_list = [], [], []

    with torch.no_grad():
        for i, (imgs, lbls, paths) in enumerate(loader):
            imgs  = imgs.to(device)
            feats = model(imgs)                   # (B, 384)
            features_list.append(feats.cpu().numpy())
            labels_list.append(lbls.numpy())
            paths_list.extend(paths)
            if (i + 1) % 10 == 0:
                print(f"  batch {i + 1}/{len(loader)}")

    features = np.concatenate(features_list)     # (N, 384)
    labels   = np.concatenate(labels_list)       # (N,)
    paths    = np.array(paths_list)              # (N,) strings

    print(f"Feature matrix shape: {features.shape}")

    # ---------------------------------------------------------------------- #
    # Fixed val split — stratified, never corrupted
    # ---------------------------------------------------------------------- #
    rng = np.random.default_rng(SEED)
    val_indices = []
    for cls_idx in range(len(CLASSES)):
        cls_mask = np.where(labels == cls_idx)[0]
        n_val    = max(1, int(len(cls_mask) * VAL_FRAC))
        chosen   = rng.choice(cls_mask, size=n_val, replace=False)
        val_indices.extend(chosen.tolist())
    val_indices = np.array(sorted(val_indices))

    # ---------------------------------------------------------------------- #
    # Save
    # ---------------------------------------------------------------------- #
    np.save(BASE_DIR / 'features.npy',    features)
    np.save(BASE_DIR / 'labels.npy',      labels)
    np.save(BASE_DIR / 'image_paths.npy', paths)
    np.save(BASE_DIR / 'val_indices.npy', val_indices)

    train_indices = np.setdiff1d(np.arange(len(labels)), val_indices)
    print(f"\nSaved:")
    print(f"  features.npy    {features.shape}")
    print(f"  labels.npy      {labels.shape}")
    print(f"  image_paths.npy {paths.shape}")
    print(f"  val_indices.npy {val_indices.shape}  ({len(val_indices)} val / {len(train_indices)} train)")

    class_counts = {CLASSES[i]: int((labels == i).sum()) for i in range(len(CLASSES))}
    print(f"\nClass distribution: {class_counts}")

    return features, labels, paths, val_indices


if __name__ == '__main__':
    extract_dino_features()
