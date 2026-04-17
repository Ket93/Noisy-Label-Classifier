"""
Figure 6 — Learning Curves (Val accuracy over epochs, 40% uniform noise).

One line per method, at fixed 40% uniform noise rate.

Reads from: results/synthetic_aux.json
"""

import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

AUX_FILE    = ROOT / 'results' / 'synthetic_aux.json'
OUTPUT_FILE = ROOT / 'results' / 'fig6_learning_curves.png'

METHOD_ORDER = ['BaselineCE', 'LabelSmoothing', 'SCE', 'GCE',
                'GMMReweight', 'ConfidentLearning']


def plot():
    with open(AUX_FILE) as f:
        aux = json.load(f)

    curves = aux.get('epoch_curves', {})
    if not curves:
        print("No epoch curves in synthetic_aux.json. Run the sweep first.")
        return

    colors = cm.tab10(np.linspace(0, 0.8, len(METHOD_ORDER)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title('Learning Curves — 40% Uniform Noise (Seed 0)', fontsize=12)

    for m_name, color in zip(METHOD_ORDER, colors):
        if m_name in curves:
            accs = curves[m_name]
            ax.plot(range(1, len(accs) + 1), accs,
                    label=m_name, color=color, linewidth=1.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_FILE}")
    plt.show()


if __name__ == '__main__':
    plot()
