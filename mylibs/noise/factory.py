import csv
import numpy as np
from pathlib import Path

_DEFAULT_ANNOTATIONS_CSV = Path(__file__).parent.parent / 'manual_annotations.csv'
_DEFAULT_IMAGE_DIR = Path(__file__).parent.parent.parent / 'images'

# ── Uniform noise ────────────────────────────────────────────────────────────

def uniform_noise(labels, noise_rate, num_classes=5, seed=42):
    rng = np.random.default_rng(seed)
    noisy = labels.copy()
    for i in range(len(labels)):
        if rng.random() < noise_rate:
            noisy[i] = rng.choice([c for c in range(num_classes) if c != labels[i]])
    return noisy


# ── Asymmetric noise ─────────────────────────────────────────────────────────

FLOWERS_FLIP_PAIRS = {
    1: 2,   # rose      → tulip
    2: 1,   # tulip     → rose
    0: 3,   # daisy     → sunflower
    3: 0,   # sunflower → daisy
    1: 0,   # dandelion → daisy
    0: 1,   # daisy     → dandelion
    1: 3,   # dandelion → sunflower
    3: 1,   # sunflower → dandelion
}

def asymmetric_noise(labels, noise_rate, flip_pairs=FLOWERS_FLIP_PAIRS, seed=42):
    rng = np.random.default_rng(seed)
    noisy = labels.copy()
    for i in range(len(labels)):
        if labels[i] in flip_pairs and rng.random() < noise_rate:
            noisy[i] = flip_pairs[labels[i]]
    return noisy


# ── Instance-dependent noise ─────────────────────────────────────────────────

def instance_dependent_noise(features, labels, base_rate, seed=42):
    rng = np.random.default_rng(seed)
    noisy = labels.copy()
    classes = np.unique(labels)
    centroids = {c: features[labels == c].mean(axis=0) for c in classes}

    for i, (feat, label) in enumerate(zip(features, labels)):
        class_feats = features[labels == label]
        centroid = centroids[label]
        dists = np.linalg.norm(class_feats - centroid, axis=1)
        dist_i = np.linalg.norm(feat - centroid)
        max_d = dists.max()

        p_noise = float(np.clip(base_rate * (dist_i / max_d) if max_d > 0 else 0.0, 0.0, 1.0))
        if rng.random() < p_noise:
            noisy[i] = rng.choice([c for c in classes if c != label])

    return noisy


# ── Human annotation noise ────────────────────────────────────────────────────

def load_annotations_csv(annotations_csv: str) -> dict:
    mapping = {}
    csv_path = Path(annotations_csv)
    if not csv_path.exists():
        return mapping
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            label = int(row['label'])
            if label == -1:
                continue
            rel = row['filename'].replace('\\', '/')
            mapping[rel] = label
    return mapping


def human_noise(
    labels: np.ndarray,
    image_paths: np.ndarray,
    train_idx: np.ndarray,
    annotations_csv: str = None,
    image_dir: str = None,
) -> np.ndarray:
    if annotations_csv is None:
        annotations_csv = str(_DEFAULT_ANNOTATIONS_CSV)
    if image_dir is None:
        image_dir = str(_DEFAULT_IMAGE_DIR)

    annotation_map = load_annotations_csv(annotations_csv)
    image_dir = Path(image_dir)

    noisy = labels[train_idx].copy()
    n_swaps = 0

    for arr_pos, global_idx in enumerate(train_idx):
        abs_path = Path(str(image_paths[global_idx]))
        try:
            rel = abs_path.relative_to(image_dir).as_posix()
        except ValueError:
            continue

        if rel in annotation_map:
            noisy[arr_pos] = annotation_map[rel]
            n_swaps += 1

    return noisy


# ── Factory ───────────────────────────────────────────────────────────────────

def apply_noise(features, labels, noise_type, noise_rate=0.0, seed=42, **kwargs):
    noise_type = noise_type.lower()

    if noise_type == 'uniform':
        return uniform_noise(labels, noise_rate,
                             num_classes=kwargs.get('num_classes', 5),
                             seed=seed)
    elif noise_type == 'asymmetric':
        extra = {k: kwargs[k] for k in ('flip_pairs',) if k in kwargs}
        return asymmetric_noise(labels, noise_rate, seed=seed, **extra)

    elif noise_type == 'instance':
        return instance_dependent_noise(features, labels, base_rate=noise_rate, seed=seed)

    elif noise_type == 'human':
        return human_noise(
            labels,
            kwargs['image_paths'],
            kwargs['train_idx'],
            annotations_csv=kwargs.get('annotations_csv', None),
            image_dir=kwargs.get('image_dir', None),
        )
    else:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. "
            "Choose from: 'uniform', 'asymmetric', 'instance', 'human'."
        )
