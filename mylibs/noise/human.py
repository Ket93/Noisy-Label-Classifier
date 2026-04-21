"""
Human annotation noise — Phase 2 addition.

Loads manual_annotations.csv produced by the Flask annotation tool and
builds a noisy label array for the training split.

The CSV schema (from annotationTool.py):
    filename    : path relative to IMAGE_DIR, e.g. "rose/img_001.jpg"
    label       : annotator's class index (0-4), or -1 if skipped/timed-out
    time_seconds: time taken to annotate (float)
    true_class  : folder name (ground-truth class string)

Samples not covered by annotations (or skipped) keep their clean label,
so this always returns a full-length array for the training indices.
"""

import csv
import numpy as np
from pathlib import Path

CLASSES    = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
_CLASS_MAP = {name: idx for idx, name in enumerate(CLASSES)}


def _load_csv(annotations_csv: str) -> dict:
    """Returns {relative_path: annotated_label_int} for non-skipped rows."""
    mapping = {}
    csv_path = Path(annotations_csv)
    if not csv_path.exists():
        return mapping
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            label = int(row['label'])
            if label == -1:          # skipped / timed-out
                continue
            # Normalise path separators
            rel = row['filename'].replace('\\', '/')
            mapping[rel] = label
    return mapping


def human_noise(
    labels: np.ndarray,
    image_paths: np.ndarray,
    train_idx: np.ndarray,
    annotations_csv: str = str(Path(__file__).parent.parent / 'manual_annotations.csv'),
    image_dir: str = None,
) -> np.ndarray:
    """
    Build a noisy label array using human annotations from the Flask tool.

    Parameters
    ----------
    labels          : clean ground-truth labels, shape (N,)
    image_paths     : absolute path strings saved by extract_features.py, shape (N,)
    train_idx       : indices of training samples (annotations only apply here)
    annotations_csv : path to the CSV written by annotationTool.py
    image_dir       : base IMAGE_DIR used to compute relative paths;
                      if None, derived from the first image_path entry

    Returns
    -------
    noisy : np.ndarray shape (len(train_idx),)
        Labels for training samples, with human annotations substituted where
        available.  Unannotated / skipped samples retain clean labels.
    """
    annotation_map = _load_csv(annotations_csv)

    if len(annotation_map) == 0:
        print("[human_noise] No annotations found — returning clean labels.")
        return labels[train_idx].copy()

    # Determine image_dir so we can compute relative paths
    if image_dir is None:
        # image_paths[0] looks like ".../Noisy-Label-Classifier/images/rose/img.jpg"
        # We want the parent of the class folders, i.e. the "images/" dir.
        sample_path = Path(str(image_paths[train_idx[0]]))
        # Go up one level from the class folder: rose/ -> images/
        image_dir = str(sample_path.parent.parent)

    image_dir = Path(image_dir)

    noisy   = labels[train_idx].copy()
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

    total      = len(train_idx)
    noise_rate = (noisy != labels[train_idx]).mean()

    return noisy


def human_noise_rate(
    labels: np.ndarray,
    image_paths: np.ndarray,
    train_idx: np.ndarray,
    annotations_csv: str = str(Path(__file__).parent.parent / 'manual_annotations.csv'),
    image_dir: str = None,
) -> float:
    """Convenience: return the fraction of annotated training samples that are wrong."""
    noisy = human_noise(labels, image_paths, train_idx,
                        annotations_csv=annotations_csv, image_dir=image_dir)
    return float((noisy != labels[train_idx]).mean())
