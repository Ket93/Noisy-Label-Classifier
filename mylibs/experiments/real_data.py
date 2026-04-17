"""
Phase 4B — Real human annotation noise experiment.

Uses the manual_annotations.csv produced by the Flask annotation tool
(mylibs/annotations/annotationTool.py) as a source of real human-generated
noisy labels, analogous to CIFAR-10N in the outline.

Label sets evaluated:
  clean        — original ground-truth labels (upper bound)
  human        — human annotations from the Flask tool (real noise)
  uniform_30   — 30% synthetic uniform noise (reference point)
  asymmetric_30 — 30% synthetic asymmetric noise (reference point)

Results saved to results/real_results.json.

Usage:
    cd Noisy-Label-Classifier
    python -m experiments.real_data

Note: run the Flask annotation tool first and collect ≥100 annotations
to make the human noise condition meaningful.
"""

import sys, json, time
from pathlib import Path

ROOT = Path(__file__).parent.parent   # mylibs/
sys.path.insert(0, str(ROOT))

import numpy as np

from dataset import load_data, CLASSES
from noise.factory  import apply_noise
from methods import (BaselineCE, LabelSmoothing, SCE, GCE,
                     SmallLoss, GMMReweight, ConfidentLearning, Curriculum)

RESULTS_DIR       = ROOT / 'results'
ANNOTATIONS_CSV   = ROOT / 'manual_annotations.csv'
EPOCHS            = 100
BATCH_SIZE        = 256
RESULTS_DIR.mkdir(exist_ok=True)


def make_methods(val_features, val_labels):
    kw = dict(val_features=val_features, val_labels=val_labels, epochs=EPOCHS)
    return [
        BaselineCE(**kw),
        LabelSmoothing(**kw),
        SCE(**kw),
        GCE(**kw),
        SmallLoss(**kw),
        GMMReweight(warmup_epochs=30, total_epochs=EPOCHS, **kw),
        ConfidentLearning(warmup_epochs=EPOCHS // 2, total_epochs=EPOCHS, **kw),
        Curriculum(**kw),
    ]


def run():
    features, labels, image_paths, train_idx, val_idx = load_data(ROOT)

    train_features = features[train_idx]
    train_labels   = labels[train_idx]
    val_features   = features[val_idx]
    val_labels     = labels[val_idx]

    # ---------------------------------------------------------------------- #
    # Build label sets
    # ---------------------------------------------------------------------- #
    label_sets = {}

    # 1) Clean (ground truth) — upper bound
    label_sets['clean'] = train_labels.copy()

    # 2) Human annotations from Flask tool
    human_noisy = apply_noise(
        train_features, labels, 'human',
        image_paths=image_paths,
        train_idx=train_idx,
        annotations_csv=str(ANNOTATIONS_CSV),
    )
    label_sets['human'] = human_noisy

    # 3) Synthetic reference points
    label_sets['uniform_30'] = apply_noise(
        train_features, train_labels, 'uniform', 0.3, seed=42)
    label_sets['asymmetric_30'] = apply_noise(
        train_features, train_labels, 'asymmetric', 0.3, seed=42)

    # Actual noise rates for reporting
    noise_rates_info = {}
    for name, noisy in label_sets.items():
        rate = float((noisy != train_labels).mean())
        noise_rates_info[name] = rate
        print(f"  {name}: noise_rate = {rate:.3f}")

    # ---------------------------------------------------------------------- #
    # Evaluate all methods × label sets
    # ---------------------------------------------------------------------- #
    real_results = {ls: {} for ls in label_sets}

    t0 = time.time()
    total = len(label_sets) * 8
    count = 0

    for ls_name, noisy_labels in label_sets.items():
        methods = make_methods(val_features, val_labels)
        for method in methods:
            count += 1
            print(f"[{count}/{total}] {ls_name} / {method.name} ...", end='', flush=True)
            method.fit(train_features, noisy_labels)
            acc = method.evaluate(val_features, val_labels)
            print(f" acc={acc:.4f}")
            real_results[ls_name][method.name] = acc

    # ---------------------------------------------------------------------- #
    # Save
    # ---------------------------------------------------------------------- #
    output = {
        'results':     real_results,
        'noise_rates': noise_rates_info,
        'classes':     CLASSES,
        'label_sets':  list(label_sets.keys()),
    }
    with open(RESULTS_DIR / 'real_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print("Saved: results/real_results.json")


if __name__ == '__main__':
    run()
