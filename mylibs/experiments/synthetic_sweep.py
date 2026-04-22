import sys, json, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from dataset import load_data, CLASSES
from noise.factory  import apply_noise
from methods import (BaselineCE, LabelSmoothing, SCE, GCE,
                     GMMReweight, ConfidentLearning)

# config
NOISE_TYPES  = ['uniform', 'asymmetric', 'instance']
NOISE_RATES  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_SEEDS      = 3
EPOCHS       = 100
BATCH_SIZE   = 256

# conditions for auxiliary data collection
AUX_NOISE_TYPE_HIST  = 'uniform'
AUX_NOISE_RATE_CL    = 0.3
AUX_NOISE_RATE_CURVE = 0.4

RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# helpers
def make_methods(val_features, val_labels):
    """Return a fresh list of method instances for one experiment run."""
    kw = dict(val_features=val_features, val_labels=val_labels, epochs=EPOCHS)
    return [
        BaselineCE(**kw),
        LabelSmoothing(**kw),
        SCE(**kw),
        GCE(**kw),
        GMMReweight(warmup_epochs=30, total_epochs=EPOCHS, **kw),
        ConfidentLearning(warmup_epochs=EPOCHS // 2, total_epochs=EPOCHS, **kw),
    ]

def nested_dict():
    from collections import defaultdict
    return defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

def run():
    features, labels, image_paths, train_idx, val_idx = load_data(ROOT)

    train_features = features[train_idx]
    train_labels = labels[train_idx]
    val_features = features[val_idx]
    val_labels = labels[val_idx]

    # store results as plain dicts for JSON serialisation
    results = {}
    for nt in NOISE_TYPES:
        results[nt] = {}
        for nr in NOISE_RATES:
            results[nt][str(nr)] = {}

    # auxiliary collections
    aux_losses = {} # (noise_type, noise_rate, method_name) -> list of losses (seed 0 only)
    aux_cl_matrix = None # estimated count matrix at 30% uniform, seed 0
    aux_cl_true_matrix = None
    aux_cl_flagged = None # flagged indices at 30% uniform, seed 0
    aux_cl_noise_est = {} # (noise_type, noise_rate) -> estimated noise rate (seed 0)
    aux_epoch_curves = {} # method_name -> epoch val accs at 40% uniform, seed 0

    total_runs = len(NOISE_TYPES) * len(NOISE_RATES) * N_SEEDS
    run_count  = 0
    t0 = time.time()

    for noise_type in NOISE_TYPES:
        for noise_rate in NOISE_RATES:
            for seed in range(N_SEEDS):
                run_count += 1

                # apply noise to training labels only
                if noise_type in ('uniform', 'asymmetric'):
                    noisy_train = apply_noise(
                        train_features, train_labels, noise_type, noise_rate, seed=seed)
                else:  # 'instance'
                    noisy_train = apply_noise(
                        train_features, train_labels, noise_type, noise_rate, seed=seed)

                methods = make_methods(val_features, val_labels)

                for method in methods:
                    print(f"    {method.name} ...", end='', flush=True)
                    method.fit(train_features, noisy_train)
                    acc = method.evaluate(val_features, val_labels)
                    print(f" acc={acc:.4f}")

                    nr_key = str(noise_rate)
                    if method.name not in results[noise_type][nr_key]:
                        results[noise_type][nr_key][method.name] = []
                    results[noise_type][nr_key][method.name].append(acc)

                    # ----- Auxiliary data (seed 0 only) -----
                    if seed == 0:
                        # per-sample losses for all conditions (GMM histogram)
                        if method.per_sample_losses is not None:
                            key = f"{noise_type}_{noise_rate}_{method.name}"
                            aux_losses[key] = method.per_sample_losses.tolist()

                        # learning curves at 40% uniform (Figure 6)
                        if (noise_type == 'uniform' and noise_rate == AUX_NOISE_RATE_CURVE
                                and method.epoch_val_accs):
                            aux_epoch_curves[method.name] = method.epoch_val_accs

                        # CL-specific extras at 30% uniform (Figures 3, 4, 5)
                        if (isinstance(method, ConfidentLearning)
                                and noise_type == AUX_NOISE_TYPE_HIST
                                and noise_rate == AUX_NOISE_RATE_CL):
                            aux_cl_matrix  = method.count_matrix.tolist()
                            aux_cl_flagged = method.flagged_indices.tolist()

                            # true count matrix from actual label flips
                            K = len(CLASSES)
                            true_matrix = np.zeros((K, K), dtype=np.int32)
                            for i in range(len(noisy_train)):
                                true_matrix[train_labels[i], noisy_train[i]] += 1
                            aux_cl_true_matrix = true_matrix.tolist()

                        # CL noise rate estimates for all conditions (Figure 5)
                        if isinstance(method, ConfidentLearning):
                            key = f"{noise_type}_{noise_rate}"
                            aux_cl_noise_est[key] = method.estimated_noise_rate

    # store data
    with open(RESULTS_DIR / 'synthetic_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    aux = {
        'losses':          aux_losses,
        'cl_count_matrix': aux_cl_matrix,
        'cl_true_matrix':  aux_cl_true_matrix,
        'cl_flagged':      aux_cl_flagged,
        'cl_noise_est':    aux_cl_noise_est,
        'epoch_curves':    aux_epoch_curves,
        'noise_rates':     NOISE_RATES,
        'noise_types':     NOISE_TYPES,
        'classes':         CLASSES,
    }
    with open(RESULTS_DIR / 'synthetic_aux.json', 'w') as f:
        json.dump(aux, f, indent=2)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"Saved: results/synthetic_results.json")
    print(f"Saved: results/synthetic_aux.json")


if __name__ == '__main__':
    run()
