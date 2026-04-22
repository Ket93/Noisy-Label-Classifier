# figure 3

import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

AUX_FILE = ROOT / 'results' / 'synthetic_aux.json'
OUTPUT_FILE = ROOT / 'results' / 'fig3_transition_matrix.png'


def _plot_matrix(ax, mat, classes, title, vmax=None):
    mat = np.array(mat, dtype=float)
    im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=vmax or mat.max())
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel('Assigned Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    for i in range(len(classes)):
        for j in range(len(classes)):
            val = int(mat[i, j])
            color = 'white' if mat[i, j] > mat.max() * 0.6 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=7, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot():
    with open(AUX_FILE) as f:
        aux = json.load(f)

    classes = aux.get('classes', [f'C{i}' for i in range(5)])
    true_mat = aux.get('cl_true_matrix')
    est_mat = aux.get('cl_count_matrix')

    if true_mat is None or est_mat is None:
        print("Transition matrix data not found in synthetic_aux.json. "
              "Run experiments/synthetic_sweep.py first.")
        return

    vmax = max(np.array(true_mat).max(), np.array(est_mat).max())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Noise Transition Matrix at 30% Uniform Noise', fontsize=12)

    _plot_matrix(ax1, true_mat, classes, 'Ground Truth')
    _plot_matrix(ax2, est_mat,  classes, "Confident Learning's Estimate", vmax=vmax)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_FILE}")
    plt.show()


if __name__ == '__main__':
    plot()
