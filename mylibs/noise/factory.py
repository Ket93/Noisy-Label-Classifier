"""
Single entry point for all noise types.

Usage
-----
    from noise.factory import apply_noise

    # Synthetic
    noisy = apply_noise(features, labels, 'uniform',     noise_rate=0.3)
    noisy = apply_noise(features, labels, 'asymmetric',  noise_rate=0.3)
    noisy = apply_noise(features, labels, 'instance',    noise_rate=0.3)

    # Human (requires image_paths and train_idx)
    noisy = apply_noise(
        features, labels, 'human',
        image_paths=image_paths,
        train_idx=train_idx,
        annotations_csv='manual_annotations.csv',
    )

Note: for 'human' the returned array has shape (len(train_idx),), not (N,).
For all other types the full labels array (N,) is returned.
"""

import numpy as np
from pathlib import Path
from .uniform            import uniform_noise
from .asymmetric         import asymmetric_noise
from .instance_dependent import instance_dependent_noise
from .human              import human_noise

NUM_CLASSES = 5   # daisy, dandelion, rose, sunflower, tulip


def apply_noise(
    features:   np.ndarray,
    labels:     np.ndarray,
    noise_type: str,
    noise_rate: float = 0.0,
    seed:       int   = 42,
    **kwargs,
) -> np.ndarray:
    """
    Apply the requested noise type and return the noisy label array.

    Parameters
    ----------
    features   : (N, D) DINO embeddings (only used by 'instance')
    labels     : (N,)  clean labels
    noise_type : one of 'uniform', 'asymmetric', 'instance', 'human'
    noise_rate : corruption probability (ignored for 'human')
    seed       : RNG seed (ignored for 'human')
    **kwargs   : extra args forwarded to the underlying function
                 - 'human' needs: image_paths, train_idx
                   optionally:   annotations_csv, image_dir
    """
    noise_type = noise_type.lower()

    if noise_type == 'uniform':
        return uniform_noise(labels, noise_rate,
                             num_classes=kwargs.get('num_classes', NUM_CLASSES),
                             seed=seed)

    elif noise_type == 'asymmetric':
        extra = {}
        if 'flip_pairs' in kwargs:
            extra['flip_pairs'] = kwargs['flip_pairs']
        return asymmetric_noise(labels, noise_rate, seed=seed, **extra)

    elif noise_type == 'instance':
        return instance_dependent_noise(features, labels, base_rate=noise_rate, seed=seed)

    elif noise_type == 'human':
        default_csv = str(Path(__file__).parent.parent / 'manual_annotations.csv')
        return human_noise(
            labels,
            kwargs['image_paths'],
            kwargs['train_idx'],
            annotations_csv=kwargs.get('annotations_csv', default_csv),
            image_dir=kwargs.get('image_dir', None),
        )

    else:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. "
            "Choose from: 'uniform', 'asymmetric', 'instance', 'human'."
        )
