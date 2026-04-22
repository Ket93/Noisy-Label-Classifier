import sys, json, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from dataset import load_data, CLASSES
from noise.factory  import apply_noise
from methods import (BaselineCE, LabelSmoothing, SCE, GCE,
                     GMMReweight, ConfidentLearning)

RESULTS_DIR = ROOT / 'results'
ANNOTATIONS_CSV = ROOT / 'manual_annotations.csv'
EPOCHS = 100
BATCH_SIZE = 256
RESULTS_DIR.mkdir(exist_ok=True)


def make_methods(val_features, val_labels):
    kw = dict(val_features=val_features, val_labels=val_labels, epochs=EPOCHS)
    return [
        BaselineCE(**kw),
        LabelSmoothing(**kw),
        SCE(**kw),
        GCE(**kw),
        GMMReweight(warmup_epochs=30, total_epochs=EPOCHS, **kw),
        ConfidentLearning(warmup_epochs=EPOCHS // 2, total_epochs=EPOCHS, **kw),
    ]


def run():
    features, labels, image_paths, train_idx, val_idx = load_data(ROOT)

    train_features = features[train_idx]
    train_labels = labels[train_idx]
    val_features = features[val_idx]
    val_labels = labels[val_idx]
    label_sets = {}

    label_sets['clean'] = train_labels.copy()

    # annotations from tool
    human_noisy = apply_noise(
        train_features, labels, 'human',
        image_paths=image_paths,
        train_idx=train_idx,
        annotations_csv=str(ANNOTATIONS_CSV),
    )
    label_sets['human'] = human_noisy

    label_sets['uniform_30'] = apply_noise(
        train_features, train_labels, 'uniform', 0.3, seed=42)
    label_sets['asymmetric_30'] = apply_noise(
        train_features, train_labels, 'asymmetric', 0.3, seed=42)

    noise_rates_info = {}
    for name, noisy in label_sets.items():
        rate = float((noisy != train_labels).mean())
        noise_rates_info[name] = rate
        print(f"  {name}: noise_rate = {rate:.3f}")

    # evaluate methods
    real_results = {ls: {} for ls in label_sets}

    t0 = time.time()
    total = len(label_sets) * 6
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

    # store data
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
