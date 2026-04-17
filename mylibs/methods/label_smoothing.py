import numpy as np
import torch
import torch.nn.functional as F
from .base     import BaseMethod
from ._trainer import (make_model, get_device, train_standard,
                       predict_from_model, predict_proba_from_model,
                       per_sample_ce_loss, EPOCHS, BATCH_SIZE, LR, NUM_CLASSES)


def _smooth_ce(logits, targets, idx, epsilon: float, num_classes: int):
    """Cross-entropy with label smoothing applied inline."""
    log_probs = F.log_softmax(logits, dim=1)

    # Soft targets: (1-eps) * one_hot + eps/K
    one_hot  = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)
    smooth   = (1.0 - epsilon) * one_hot + epsilon / num_classes

    loss = -(smooth * log_probs).sum(dim=1).mean()
    return loss


class LabelSmoothing(BaseMethod):
    """
    Cross-entropy with label smoothing.
    Prevents overconfidence on noisy labels by distributing a small
    probability mass uniformly across all classes.
    """
    name = 'LabelSmoothing'

    def __init__(self, epsilon=0.1, num_classes=NUM_CLASSES,
                 epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                 val_features=None, val_labels=None):
        self.epsilon      = epsilon
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

    def fit(self, features: np.ndarray, noisy_labels: np.ndarray) -> 'LabelSmoothing':
        eps = self.epsilon
        K   = self.num_classes

        def loss_fn(logits, targets, idx):
            return _smooth_ce(logits, targets, idx, eps, K)

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
