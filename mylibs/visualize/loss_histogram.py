# figure 2

import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent   # mylibs/
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from dataset import load_data, CLASSES
from noise.factory  import apply_noise

AUX_FILE    = ROOT / 'results' / 'synthetic_aux.json'
OUTPUT_FILE = ROOT / 'results' / 'fig2_loss_histogram.png'
NOISE_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]


def plot():
    with open(AUX_FILE) as f:
        aux = json.load(f)

    features, labels, image_paths, train_idx, val_idx = load_data(ROOT)
    train_labels = labels[train_idx]
    train_features = features[train_idx]

    n_rates = len(NOISE_RATES)
    fig, axes = plt.subplots(1, n_rates, figsize=(4 * n_rates, 4))
    fig.suptitle('Per-Sample Loss Distribution (Uniform Noise, BaselineCE)', fontsize=12)

    if n_rates == 1:
        axes = [axes]

    for ax, nr in zip(axes, NOISE_RATES):
        key = f"uniform_{nr}_BaselineCE"
        losses_list = aux.get('losses', {}).get(key, None)
        if losses_list is None:
            ax.set_title(f'nr={nr}\n(no data)')
            continue

        losses = np.array(losses_list)

        # Reconstruct noisy labels (seed=0) to determine clean vs noisy
        noisy_train = apply_noise(train_features, train_labels, 'uniform', nr, seed=0)
        is_noisy    = (noisy_train != train_labels).astype(bool)

        clean_losses = losses[~is_noisy]
        noisy_losses = losses[is_noisy]

        bins = np.linspace(0, min(losses.max(), 8.0), 60)
        ax.hist(clean_losses, bins=bins, alpha=0.6, color='steelblue',  label='Clean',  density=True)
        ax.hist(noisy_losses, bins=bins, alpha=0.6, color='salmon',     label='Noisy',  density=True)

        # Fit GMM and overlay components
        if len(losses) > 10:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(losses.reshape(-1, 1))
            x_range = np.linspace(0, min(losses.max(), 8.0), 300).reshape(-1, 1)
            from scipy.stats import norm
            for comp in range(2):
                mu  = gmm.means_[comp, 0]
                sig = np.sqrt(gmm.covariances_[comp, 0, 0])
                wt  = gmm.weights_[comp]
                y   = wt * norm.pdf(x_range.ravel(), mu, sig)
                ax.plot(x_range.ravel(), y, 'k--', linewidth=1.5, alpha=0.8)

        ax.set_title(f'Noise Rate = {int(nr*100)}%')
        ax.set_xlabel('CE Loss')
        if ax == axes[0]:
            ax.set_ylabel('Density')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_FILE}")
    plt.show()


if __name__ == '__main__':
    plot()
