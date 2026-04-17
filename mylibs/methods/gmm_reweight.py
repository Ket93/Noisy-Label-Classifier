"""
GMM-based sample reweighting (inspired by DivideMix / MentorNet).

Algorithm:
  1. Train for warm-up epochs with standard CE.
  2. Compute per-sample CE losses.
  3. Fit a 2-component GMM on the loss distribution.
  4. The component with lower mean = "clean" component.
     Use posterior probability of belonging to the clean component as
     per-sample weight for the remaining training.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from .base     import BaseMethod
from ._trainer import (make_model, get_device, train_standard,
                       predict_from_model, predict_proba_from_model,
                       per_sample_ce_loss, EPOCHS, BATCH_SIZE, LR, NUM_CLASSES)


class GMMReweight(BaseMethod):
    """Train with CE, then reweight samples by GMM clean-posterior."""
    name = 'GMMReweight'

    def __init__(self, warmup_epochs=30, total_epochs=None,
                 epochs=None,
                 num_classes=NUM_CLASSES, batch_size=BATCH_SIZE, lr=LR,
                 val_features=None, val_labels=None):
        if total_epochs is None:
            total_epochs = epochs if epochs is not None else EPOCHS
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs
        self.num_classes    = num_classes
        self.batch_size     = batch_size
        self.lr             = lr
        self.val_features   = val_features
        self.val_labels     = val_labels
        self._model         = None
        self._device        = get_device()
        self.per_sample_losses  = None
        self.epoch_val_accs     = []
        self.epoch_losses       = []
        self.gmm_weights        = None   # shape (N,) — for visualisations

    def fit(self, features: np.ndarray, noisy_labels: np.ndarray) -> 'GMMReweight':
        device = self._device
        model  = make_model(self.num_classes).to(device)

        # Phase 1: warm-up with standard CE
        el, va = train_standard(
            model, features, noisy_labels,
            epochs=self.warmup_epochs, batch_size=self.batch_size, lr=self.lr,
            device=device, val_features=self.val_features, val_labels=self.val_labels,
        )
        self.epoch_losses.extend(el)
        self.epoch_val_accs.extend(va)

        # Phase 2: GMM on per-sample losses
        losses = per_sample_ce_loss(model, features, noisy_labels, device)
        gmm    = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(losses.reshape(-1, 1))

        # Clean component = lower-mean component
        clean_comp = int(gmm.means_.argmin())
        posteriors = gmm.predict_proba(losses.reshape(-1, 1))
        weights    = posteriors[:, clean_comp].astype(np.float32)
        weights    = np.clip(weights, 1e-3, 1.0)
        self.gmm_weights = weights

        # Phase 3: reweighted training
        remaining = self.total_epochs - self.warmup_epochs
        el, va = train_standard(
            model, features, noisy_labels,
            sample_weights=weights,
            epochs=remaining, batch_size=self.batch_size, lr=self.lr * 0.5,
            device=device, val_features=self.val_features, val_labels=self.val_labels,
        )
        self.epoch_losses.extend(el)
        self.epoch_val_accs.extend(va)

        self._model = model
        self.per_sample_losses = per_sample_ce_loss(model, features, noisy_labels, device)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return predict_from_model(self._model, features, self._device)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return predict_proba_from_model(self._model, features, self._device)
