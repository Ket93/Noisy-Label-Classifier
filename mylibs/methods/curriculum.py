"""
Curriculum Learning (Self-paced).
Bengio et al., "Curriculum Learning", ICML 2009.

Strategy:
  - Phase 1 (warmup): train on the easiest 50% of samples
    (lowest CE loss under a fresh model, i.e. highest confidence).
  - Phase 2: linearly expand the training set to 100% over the remaining epochs.

Easiness is measured by loss under the initial model — samples the model
already handles well are treated as clean/easy and prioritised early.
"""

import numpy as np
import torch
import torch.nn.functional as F
from .base     import BaseMethod
from ._trainer import (make_model, get_device, train_standard,
                       predict_from_model, predict_proba_from_model,
                       per_sample_ce_loss, EPOCHS, BATCH_SIZE, LR, NUM_CLASSES)


class Curriculum(BaseMethod):
    """Start on easiest 50%, linearly expand to 100% over training."""
    name = 'Curriculum'

    def __init__(self, start_frac=0.5, num_classes=NUM_CLASSES,
                 epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                 val_features=None, val_labels=None):
        self.start_frac   = start_frac
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

    def fit(self, features: np.ndarray, noisy_labels: np.ndarray) -> 'Curriculum':
        device = self._device
        n      = len(noisy_labels)
        model  = make_model(self.num_classes).to(device)
        opt    = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)

        X = torch.tensor(features,    dtype=torch.float32).to(device)
        Y = torch.tensor(noisy_labels, dtype=torch.long).to(device)

        # Initial difficulty ranking (before any training)
        with torch.no_grad():
            init_losses = F.cross_entropy(model(X), Y, reduction='none').cpu().numpy()

        # Sort by loss ascending: index 0 = easiest
        difficulty_order = np.argsort(init_losses)

        for epoch in range(self.epochs):
            # Linearly expand fraction from start_frac → 1.0
            frac       = self.start_frac + (1.0 - self.start_frac) * (epoch / max(self.epochs - 1, 1))
            keep_n     = max(1, int(n * frac))
            subset_idx = difficulty_order[:keep_n]

            # Shuffle subset into mini-batches
            perm    = subset_idx[np.random.permutation(keep_n)]
            perm_t  = torch.tensor(perm, dtype=torch.long)
            chunks  = perm_t.split(self.batch_size)

            model.train()
            epoch_loss = 0.0
            for chunk in chunks:
                xb = X[chunk]
                yb = Y[chunk]
                opt.zero_grad()
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

            self.epoch_losses.append(epoch_loss / max(len(chunks), 1))

            if self.val_features is not None and self.val_labels is not None:
                preds = predict_from_model(model, self.val_features, device)
                self.epoch_val_accs.append(float((preds == self.val_labels).mean()))

        self._model = model
        self.per_sample_losses = per_sample_ce_loss(model, features, noisy_labels, device)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return predict_from_model(self._model, features, self._device)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return predict_proba_from_model(self._model, features, self._device)
