"""
Figure 1 — Accuracy vs. Noise Rate (3 subplots, one per noise type).

X axis : noise rate
Y axis : validation accuracy
Lines  : one per method, shaded ± std over seeds

Reads from: results/synthetic_results.json
"""

import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

RESULTS_FILE = ROOT / 'results' / 'synthetic_results.json'
OUTPUT_FILE  = ROOT / 'results' / 'fig1_accuracy_curves.png'

NOISE_TYPES  = ['uniform', 'asymmetric', 'instance']
NOISE_LABELS = ['Uniform', 'Asymmetric', 'Instance-Dependent']
NOISE_RATES  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
METHOD_ORDER = ['BaselineCE', 'LabelSmoothing', 'SCE', 'GCE',
                'SmallLoss', 'GMMReweight', 'ConfidentLearning', 'Curriculum']


def plot():
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    colors = cm.tab10(np.linspace(0, 0.8, len(METHOD_ORDER)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle('Validation Accuracy vs. Noise Rate — Flowers Dataset', fontsize=13)

    for ax, noise_type, noise_label in zip(axes, NOISE_TYPES, NOISE_LABELS):
        nt_data = results.get(noise_type, {})

        for m_name, color in zip(METHOD_ORDER, colors):
            means, stds = [], []
            for nr in NOISE_RATES:
                accs = nt_data.get(str(float(nr)), {}).get(m_name, [])
                if accs:
                    means.append(np.mean(accs))
                    stds.append(np.std(accs))
                else:
                    means.append(np.nan)
                    stds.append(0.0)

            means = np.array(means)
            stds  = np.array(stds)
            valid = ~np.isnan(means)

            ax.plot(np.array(NOISE_RATES)[valid], means[valid],
                    label=m_name, color=color, marker='o', linewidth=1.8)
            ax.fill_between(np.array(NOISE_RATES)[valid],
                            (means - stds)[valid], (means + stds)[valid],
                            color=color, alpha=0.15)

        ax.set_title(noise_label)
        ax.set_xlabel('Noise Rate')
        ax.set_xticks(NOISE_RATES)
        ax.set_xlim(-0.02, 0.52)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.4)

    axes[0].set_ylabel('Validation Accuracy')
    axes[-1].legend(loc='lower left', fontsize=8, ncol=1)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_FILE}")
    plt.show()


if __name__ == '__main__':
    plot()
