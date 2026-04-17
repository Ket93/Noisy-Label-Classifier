"""
Small-Loss trick (Co-teaching variant, simplified to single network).
Han et al., "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels", NeurIPS 2018.

Each epoch:
  1. Compute per-sample CE loss on the current batch
  2. Keep only the R% samples with the smallest loss (assumed clean)
  3. Update on those samples only

R starts at (1 - noise_rate_estimate) * 100% and stays fixed here.
We estimate noise_rate as min(0.5, the fraction of samples with loss > median).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .base     import BaseMethod
from ._trainer import (make_model, get_device,
                       predict_from_model, predict_proba_from_model,
                       features_to_tensor, labels_to_tensor,
                       per_sample_ce_loss, EPOCHS, BATCH_SIZE, LR, NUM_CLASSES)


class SmallLoss(BaseMethod):
    """Train only on the R% lowest-loss samples per epoch."""
    name = 'SmallLoss'

    def __init__(self, keep_rate=0.7, num_classes=NUM_CLASSES,
                 epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                 val_features=None, val_labels=None):
        self.keep_rate    = keep_rate   # fraction of samples to keep
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

    def fit(self, features: np.ndarray, noisy_labels: np.ndarray) -> 'SmallLoss':
        device = self._device
        model  = make_model(self.num_classes).to(device)
        opt    = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)

        X = features_to_tensor(features).to(device)
        Y = labels_to_tensor(noisy_labels).to(device)

        keep_n = max(1, int(len(Y) * self.keep_rate))

        for epoch in range(self.epochs):
            model.train()

            # Compute all per-sample losses
            with torch.no_grad():
                logits_all = model(X)
                losses_all = F.cross_entropy(logits_all, Y, reduction='none')

            # Select keep_n indices with smallest loss
            _, keep_idx = losses_all.topk(keep_n, largest=False)

            # Mini-batch update on selected samples
            perm    = keep_idx[torch.randperm(keep_n)]
            n_batch = max(1, keep_n // self.batch_size)
            chunks  = torch.chunk(perm, n_batch)
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
