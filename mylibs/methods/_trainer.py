"""
Shared PyTorch training utilities used by all method implementations.

All methods use a single linear layer (384 -> num_classes) on top of frozen
DINO features.  This is intentionally simple — the interesting part is what
each method does with the loss, not the architecture.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

INPUT_DIM   = 384
NUM_CLASSES = 5
LR          = 1e-2
WEIGHT_DECAY = 1e-4
EPOCHS      = 100
BATCH_SIZE  = 256


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_model(num_classes: int = NUM_CLASSES) -> nn.Linear:
    return nn.Linear(INPUT_DIM, num_classes)


def features_to_tensor(features: np.ndarray) -> torch.Tensor:
    return torch.tensor(features, dtype=torch.float32)


def labels_to_tensor(labels: np.ndarray) -> torch.Tensor:
    return torch.tensor(labels, dtype=torch.long)


def predict_from_model(model: nn.Linear, features: np.ndarray, device) -> np.ndarray:
    model.eval()
    X = features_to_tensor(features).to(device)
    with torch.no_grad():
        logits = model(X)
    return logits.argmax(dim=1).cpu().numpy()


def predict_proba_from_model(model: nn.Linear, features: np.ndarray, device) -> np.ndarray:
    model.eval()
    X = features_to_tensor(features).to(device)
    with torch.no_grad():
        probs = F.softmax(model(X), dim=1)
    return probs.cpu().numpy()


def per_sample_ce_loss(model: nn.Linear, features: np.ndarray, labels: np.ndarray, device) -> np.ndarray:
    """Return per-sample cross-entropy losses without gradient computation."""
    model.eval()
    X = features_to_tensor(features).to(device)
    Y = labels_to_tensor(labels).to(device)
    with torch.no_grad():
        logits = model(X)
        losses = F.cross_entropy(logits, Y, reduction='none')
    return losses.cpu().numpy()


def train_standard(
    model:       nn.Linear,
    features:    np.ndarray,
    labels:      np.ndarray,
    sample_weights: np.ndarray = None,
    epochs:      int   = EPOCHS,
    batch_size:  int   = BATCH_SIZE,
    lr:          float = LR,
    weight_decay: float = WEIGHT_DECAY,
    device=None,
    val_features: np.ndarray = None,
    val_labels:   np.ndarray = None,
    loss_fn=None,
) -> tuple:
    """
    Generic training loop.

    Returns
    -------
    epoch_losses    : list[float]  mean train loss per epoch
    epoch_val_accs  : list[float]  val accuracy per epoch (empty if no val set)
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X = features_to_tensor(features)
    Y = labels_to_tensor(labels)
    ds = TensorDataset(X, Y, torch.arange(len(Y)))

    if sample_weights is not None:
        w = torch.tensor(sample_weights, dtype=torch.float32)
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        loader  = DataLoader(ds, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    if loss_fn is None:
        loss_fn = lambda logits, targets, idx: F.cross_entropy(logits, targets)

    epoch_losses   = []
    epoch_val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        for xb, yb, ib in loader:
            xb, yb, ib = xb.to(device), yb.to(device), ib.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits, yb, ib)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches  += 1
        epoch_losses.append(total_loss / max(n_batches, 1))

        if val_features is not None and val_labels is not None:
            preds = predict_from_model(model, val_features, device)
            acc   = float((preds == val_labels).mean())
            epoch_val_accs.append(acc)

    return epoch_losses, epoch_val_accs
