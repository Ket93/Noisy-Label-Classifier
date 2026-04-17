"""
Abstract base class shared by all methods.

Every method exposes:
    fit(features, noisy_labels)   -> self
    predict(features)             -> np.ndarray of predicted class indices
    predict_proba(features)       -> np.ndarray of shape (N, num_classes)

Optional extras stored after fit():
    self.per_sample_losses  : np.ndarray (N,)  -- final per-sample CE loss
    self.epoch_val_accs     : list[float]       -- val accuracy per epoch
    self.epoch_losses       : list[float]       -- mean train loss per epoch
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseMethod(ABC):
    name: str = 'base'

    @abstractmethod
    def fit(self, features: np.ndarray, noisy_labels: np.ndarray) -> 'BaseMethod':
        ...

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return class probabilities (N, C). Override if the method has a natural proba."""
        raise NotImplementedError

    def fit_predict(
        self,
        train_features: np.ndarray,
        noisy_labels:   np.ndarray,
        val_features:   np.ndarray,
    ) -> np.ndarray:
        self.fit(train_features, noisy_labels)
        return self.predict(val_features)

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> float:
        preds = self.predict(features)
        return float((preds == labels).mean())
