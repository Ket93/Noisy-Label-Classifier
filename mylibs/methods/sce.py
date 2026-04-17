"""
Symmetric Cross-Entropy (SCE)
Wang et al., "Symmetric Cross Entropy for Robust Learning with Noisy Labels", ICCV 2019.

Loss = alpha * CE(p, q) + beta * RCE(p, q)
where RCE (reverse CE) = CE(q, p) = -sum_k p_k * log(q_k)

The RCE term penalises predictions that are too confident about wrong classes,
adding robustness to label noise.
"""

import numpy as np
import torch
import torch.nn.functional as F
from .base     import BaseMethod
from ._trainer import (make_model, get_device, train_standard,
                       predict_from_model, predict_proba_from_model,
                       per_sample_ce_loss, EPOCHS, BATCH_SIZE, LR, NUM_CLASSES)

_EPS = 1e-7   # numerical stability for log(0)


def _sce_loss(logits, targets, idx, alpha: float, beta: float, num_classes: int):
    probs    = F.softmax(logits, dim=1)
    one_hot  = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)

    # Forward CE
    ce = F.cross_entropy(logits, targets)

    # Reverse CE: -sum_k p(y=k|x) * log(one_hot_k + eps)
    # one_hot is 0/1; log(0) clamped to log(eps)
    log_oh  = torch.log(one_hot + _EPS)
    rce     = -(probs * log_oh).sum(dim=1).mean()

    return alpha * ce + beta * rce


class SCE(BaseMethod):
    """Symmetric Cross-Entropy."""
    name = 'SCE'

    def __init__(self, alpha=0.1, beta=1.0, num_classes=NUM_CLASSES,
                 epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                 val_features=None, val_labels=None):
        self.alpha        = alpha
        self.beta         = beta
        self.num_classes  = num_classes
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.val_features = val_features
        self.val_labels   = val_labels
        self._model       = None
        self._device      = get_device()
        self.per_sample_losses = None
        self.epoch_val_accs    = []
        self.epoch_losses      = []

    def fit(self, features: np.ndarray, noisy_labels: np.ndarray) -> 'SCE':
        a, b, K = self.alpha, self.beta, self.num_classes

        def loss_fn(logits, targets, idx):
            return _sce_loss(logits, targets, idx, a, b, K)

        self._model = make_model(self.num_classes)
        self.epoch_losses, self.epoch_val_accs = train_standard(
            self._model, features, noisy_labels,
            epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
            device=self._device,
            val_features=self.val_features, val_labels=self.val_labels,
            loss_fn=loss_fn,
        )
        self.per_sample_losses = per_sample_ce_loss(
            self._model, features, noisy_labels, self._device)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return predict_from_model(self._model, features, self._device)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return predict_proba_from_model(self._model, features, self._device)
