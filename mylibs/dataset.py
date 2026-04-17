"""
Phase 1: Dataset wrapper that loads cached DINO features + labels.
All experiments import from here — never touch raw images again after extraction.
"""

import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

BASE_DIR = Path(__file__).parent.parent
CLASSES  = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


class FeaturesDataset(Dataset):
    """Thin wrapper around cached features for use with DataLoader."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        assert len(features) == len(labels)
        self.features = features.astype(np.float32)
        self.labels   = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_data(base_dir=BASE_DIR):
    """
    Load cached features, labels, image paths, and train/val split indices.

    Returns
    -------
    features     : np.ndarray  (N, 384)
    labels       : np.ndarray  (N,)   -- clean ground-truth labels
    image_paths  : np.ndarray  (N,)   -- absolute path strings
    train_idx    : np.ndarray  -- indices of training samples
    val_idx      : np.ndarray  -- indices of held-out validation samples
    """
    base_dir = Path(base_dir)

    features    = np.load(base_dir / 'features.npy')
    labels      = np.load(base_dir / 'labels.npy')
    image_paths = np.load(base_dir / 'image_paths.npy', allow_pickle=True)
    val_idx     = np.load(base_dir / 'val_indices.npy')
    train_idx   = np.setdiff1d(np.arange(len(labels)), val_idx)

    return features, labels, image_paths, train_idx, val_idx
