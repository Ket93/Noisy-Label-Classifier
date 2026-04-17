import numpy as np


def instance_dependent_noise(
    features: np.ndarray,
    labels: np.ndarray,
    base_rate: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Instance-dependent noise: samples far from their class centroid
    (in DINO feature space) are mislabeled at higher rates.

    Realistic — annotators struggle with ambiguous examples that lie
    far from the prototypical appearance of their class.
    """
    rng      = np.random.default_rng(seed)
    noisy    = labels.copy()
    classes  = np.unique(labels)
    centroids = {c: features[labels == c].mean(axis=0) for c in classes}

    for i, (feat, label) in enumerate(zip(features, labels)):
        class_feats = features[labels == label]
        centroid    = centroids[label]

        dists   = np.linalg.norm(class_feats - centroid, axis=1)
        dist_i  = np.linalg.norm(feat - centroid)
        max_d   = dists.max()

        # Noise probability scales linearly with relative distance to centroid
        p_noise = base_rate * (dist_i / max_d) if max_d > 0 else 0.0
        p_noise = float(np.clip(p_noise, 0.0, 1.0))

        if rng.random() < p_noise:
            other    = [c for c in classes if c != label]
            noisy[i] = rng.choice(other)

    return noisy
