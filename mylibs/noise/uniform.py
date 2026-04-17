import numpy as np


def uniform_noise(labels: np.ndarray, noise_rate: float, num_classes: int, seed: int = 42) -> np.ndarray:
    """
    Symmetric noise: each label independently flips to a uniformly random
    wrong class with probability `noise_rate`.
    """
    rng   = np.random.default_rng(seed)
    noisy = labels.copy()
    for i in range(len(labels)):
        if rng.random() < noise_rate:
            other    = [c for c in range(num_classes) if c != labels[i]]
            noisy[i] = rng.choice(other)
    return noisy
