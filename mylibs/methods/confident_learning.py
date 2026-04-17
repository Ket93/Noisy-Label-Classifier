"""
Confident Learning (CL)
Northcutt et al., "Confident Learning: Estimating Uncertainty in Dataset Labels", JAIR 2021.

Pipeline:
  1. Compute out-of-fold (OOF) predicted probabilities via K-fold CV.
  2. For each class j, compute per-class threshold T_j = mean P(y=j | x_i where noisy_label=j).
  3. Build count matrix C[s][y]: estimated counts of samples whose noisy label is s
     but whose true label is y (off-diagonal = label errors).
  4. For each sample i with noisy label s: flag as error if exists k≠s with P(y=k|x) > T_k.
  5. Remove flagged samples and retrain.

Also exposes:
  - self.count_matrix       : estimated noise count matrix  (num_classes, num_classes)
  - self.true_count_matrix  : set externally for visualisation (optional)
  - self.flagged_indices     : indices of samples identified as mislabeled
  - self.estimated_noise_rate : estimated fraction of noisy labels
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from .base     import BaseMethod
from ._trainer import (make_model, get_device, train_standard,
                       predict_from_model, predict_proba_from_model,
                       per_sample_ce_loss, EPOCHS, BATCH_SIZE, LR, NUM_CLASSES)


class ConfidentLearning(BaseMethod):
    """Prune label errors with Confident Learning, then retrain on clean subset."""
    name = 'ConfidentLearning'

    def __init__(self, n_folds=5, num_classes=NUM_CLASSES,
                 warmup_epochs=50, total_epochs=None,
                 epochs=None,
                 batch_size=BATCH_SIZE, lr=LR,
                 val_features=None, val_labels=None):
        if total_epochs is None:
            total_epochs = epochs if epochs is not None else EPOCHS
        self.n_folds       = n_folds
        self.num_classes   = num_classes
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.batch_size    = batch_size
        self.lr            = lr
        self.val_features  = val_features
        self.val_labels    = val_labels

        self._model              = None
        self._device             = get_device()
        self.per_sample_losses   = None
        self.epoch_val_accs      = []
        self.epoch_losses        = []
        self.count_matrix        = None
        self.true_count_matrix   = None   # set externally if available
        self.flagged_indices     = None
        self.estimated_noise_rate = None
        self.oof_probs           = None

    # ---------------------------------------------------------------------- #
    # OOF probability estimation
    # ---------------------------------------------------------------------- #
    def _compute_oof_probs(self, features, noisy_labels):
        device   = self._device
        n        = len(noisy_labels)
        oof      = np.zeros((n, self.num_classes), dtype=np.float32)
        skf      = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold_i, (train_idx, val_idx) in enumerate(skf.split(features, noisy_labels)):
            m = make_model(self.num_classes).to(device)
            train_standard(
                m, features[train_idx], noisy_labels[train_idx],
                epochs=self.warmup_epochs, batch_size=self.batch_size, lr=self.lr,
                device=device,
            )
            probs = predict_proba_from_model(m, features[val_idx], device)
            oof[val_idx] = probs

        return oof

    # ---------------------------------------------------------------------- #
    # Count matrix estimation
    # ---------------------------------------------------------------------- #
    def _estimate_count_matrix(self, oof_probs, noisy_labels):
        K = self.num_classes
        # Per-class threshold: average predicted probability for class j
        # among samples labelled j
        thresholds = np.zeros(K, dtype=np.float32)
        for j in range(K):
            mask = noisy_labels == j
            if mask.sum() > 0:
                thresholds[j] = oof_probs[mask, j].mean()
            else:
                thresholds[j] = 1.0 / K

        # Count matrix C[s][y]: how many samples have noisy=s but CL thinks true=y
        C = np.zeros((K, K), dtype=np.int32)
        for i in range(len(noisy_labels)):
            s = noisy_labels[i]
            for y in range(K):
                if oof_probs[i, y] > thresholds[y]:
                    C[s, y] += 1

        return C, thresholds

    # ---------------------------------------------------------------------- #
    # Flag likely mislabeled samples
    # ---------------------------------------------------------------------- #
    def _flag_errors(self, oof_probs, noisy_labels, thresholds):
        flagged = []
        for i in range(len(noisy_labels)):
            s = noisy_labels[i]
            for y in range(self.num_classes):
                if y != s and oof_probs[i, y] > thresholds[y]:
                    flagged.append(i)
                    break
        return np.array(flagged, dtype=np.int64)

    # ---------------------------------------------------------------------- #
    # fit
    # ---------------------------------------------------------------------- #
    def fit(self, features: np.ndarray, noisy_labels: np.ndarray) -> 'ConfidentLearning':
        device = self._device

        # Step 1 — OOF probabilities
        oof = self._compute_oof_probs(features, noisy_labels)
        self.oof_probs = oof

        # Step 2 — count matrix + thresholds
        C, thresholds = self._estimate_count_matrix(oof, noisy_labels)
        self.count_matrix = C

        # Step 3 — flag errors
        flagged = self._flag_errors(oof, noisy_labels, thresholds)
        self.flagged_indices = flagged

        n_errors = len(flagged)
        self.estimated_noise_rate = n_errors / max(len(noisy_labels), 1)

        # Step 4 — prune and retrain on clean subset
        clean_mask = np.ones(len(noisy_labels), dtype=bool)
        if len(flagged) > 0:
            clean_mask[flagged] = False

        clean_features = features[clean_mask]
        clean_labels   = noisy_labels[clean_mask]

        model = make_model(self.num_classes)
        el, va = train_standard(
            model, clean_features, clean_labels,
            epochs=self.total_epochs, batch_size=self.batch_size, lr=self.lr,
            device=device,
            val_features=self.val_features, val_labels=self.val_labels,
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
