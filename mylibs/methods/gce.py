"""
Generalized Cross-Entropy (GCE)
Zhang & Sabuncu, "Generalized Cross Entropy Loss for Noisy Labels", NeurIPS 2018.

L_q(f(x), y) = (1 - f_y(x)^q) / q

q=0  → MAE (most robust, gradient vanishes on clean samples)
q=1  → CE  (least robust)
q∈(0,1) interpolates — q=0.7 is the paper's default recommendation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from .base     import BaseMethod
from ._trainer import (make_model, get_device, train_standard,
                       predict_from_model, predict_proba_from_model,
                       per_sample_ce_loss, EPOCHS, BATCH_SIZE, LR, NUM_CLASSES)


def _gce_loss(logits, targets, idx, q: float):
    probs    = F.softmax(logits, dim=1)
    # p_y: predicted probability for the true class, shape (B,)
    p_y = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    # Truncated q-loss (clamp to avoid nan from 0^q)
    p_y  = p_y.clamp(min=1e-7)
    loss = (1.0 - p_y ** q) / q
    return loss.mean()


class GCE(BaseMethod):
    """Generalized Cross-Entropy."""
    name = 'GCE'

    def __init__(self, q=0.7, num_classes=NUM_CLASSES,
                 epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                 val_features=None, val_labels=None):
        self.q            = q
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

    def fit(self, features: np.ndarray, noisy_labels: np.ndarray) -> 'GCE':
        q = self.q

        def loss_fn(logits, targets, idx):
            return _gce_loss(logits, targets, idx, q)

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
