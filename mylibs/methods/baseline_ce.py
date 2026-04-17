import numpy as np
from .base     import BaseMethod
from ._trainer import (make_model, get_device, train_standard,
                       predict_from_model, predict_proba_from_model,
                       per_sample_ce_loss, EPOCHS, BATCH_SIZE, LR)


class BaselineCE(BaseMethod):
    """
    Standard cross-entropy with no noise handling.
    Lower bound — shows how much noise hurts an undefended classifier.
    """
    name = 'BaselineCE'

    def __init__(self, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                 val_features=None, val_labels=None):
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

    def fit(self, features: np.ndarray, noisy_labels: np.ndarray) -> 'BaselineCE':
        self._model = make_model()
        self.epoch_losses, self.epoch_val_accs = train_standard(
            self._model, features, noisy_labels,
            epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
            device=self._device,
            val_features=self.val_features, val_labels=self.val_labels,
        )
        self.per_sample_losses = per_sample_ce_loss(
            self._model, features, noisy_labels, self._device)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return predict_from_model(self._model, features, self._device)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return predict_proba_from_model(self._model, features, self._device)
