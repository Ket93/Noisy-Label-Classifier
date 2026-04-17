"""
Figure 5 — Noise Rate Calibration Scatter.

X axis: true noise rate (fraction of labels that were actually flipped)
Y axis: estimated noise rate from Confident Learning
Ideal : diagonal line y=x

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
OUTPUT_FILE = ROOT / 'results' / 'fig5_calibration.png'

NOISE_TYPES  = ['uniform', 'asymmetric', 'instance']
NOISE_LABELS = {'uniform': 'Uniform', 'asymmetric': 'Asymmetric', 'instance': 'Instance-Dep.'}
NOISE_RATES  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
COLORS       = cm.tab10(np.linspace(0, 0.6, len(NOISE_TYPES)))


def plot():
    with open(AUX_FILE) as f:
        aux = json.load(f)

    cl_est    = aux.get('cl_noise_est', {})
    noise_rates_stored = aux.get('noise_rates', NOISE_RATES)

    # For asymmetric noise, the "true" rate differs from the requested rate
    # (only confusable-pair labels are eligible). We use the requested rate
    # as a proxy — the scatter spread reveals this discrepancy.

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Confident Learning — Noise Rate Calibration", fontsize=12)

    for nt, color in zip(NOISE_TYPES, COLORS):
        true_rates, est_rates = [], []
        for nr in noise_rates_stored:
            key = f"{nt}_{nr}"
            if key in cl_est:
                true_rates.append(nr)
                est_rates.append(cl_est[key])

        if true_rates:
            ax.scatter(true_rates, est_rates, label=NOISE_LABELS[nt],
                       color=color, s=80, zorder=3)
            ax.plot(true_rates, est_rates, color=color, linewidth=1, alpha=0.5)

    # Diagonal reference
    lim = max(max(noise_rates_stored) + 0.05, 0.6)
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1.2, label='Ideal (y=x)', zorder=2)
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.set_xlabel('True Noise Rate')
    ax.set_ylabel('CL Estimated Noise Rate')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_FILE}")
    plt.show()


if __name__ == '__main__':
    plot()
