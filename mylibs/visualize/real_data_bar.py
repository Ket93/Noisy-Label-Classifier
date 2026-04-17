"""
Figure 7 — Real Data Bar Chart.

Grouped bar chart: one group per label set (clean / human / uniform_30 / asymmetric_30),
bars coloured by method.

Reads from: results/real_results.json
"""

import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

RESULTS_FILE = ROOT / 'results' / 'real_results.json'
OUTPUT_FILE  = ROOT / 'results' / 'fig7_real_data_bar.png'

METHOD_ORDER = ['BaselineCE', 'LabelSmoothing', 'SCE', 'GCE',
                'GMMReweight', 'ConfidentLearning']

LABEL_SET_DISPLAY = {
    'clean':         'Clean Labels',
    'human':         'Human Annotations',
    'uniform_30':    'Uniform 30%',
    'asymmetric_30': 'Asymmetric 30%',
}


def plot():
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    results    = data.get('results', {})
    noise_info = data.get('noise_rates', {})
    label_sets = data.get('label_sets', list(results.keys()))

    n_groups  = len(label_sets)
    n_methods = len(METHOD_ORDER)
    bar_width = 0.8 / n_methods

    colors = cm.tab10(np.linspace(0, 0.8, n_methods))

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 3), 5))
    ax.set_title('Validation Accuracy on Real Human Annotation Noise', fontsize=12)

    x_base = np.arange(n_groups)

    for m_i, (m_name, color) in enumerate(zip(METHOD_ORDER, colors)):
        accs = []
        for ls in label_sets:
            val = results.get(ls, {}).get(m_name, None)
            accs.append(val if val is not None else 0.0)

        offset = (m_i - n_methods / 2 + 0.5) * bar_width
        bars   = ax.bar(x_base + offset, accs, width=bar_width * 0.9,
                        color=color, label=m_name)

    # X-axis labels include actual noise rate
    xlabels = []
    for ls in label_sets:
        display = LABEL_SET_DISPLAY.get(ls, ls)
        rate    = noise_info.get(ls)
        if rate is not None:
            display += f'\n(nr={rate:.2f})'
        xlabels.append(display)

    ax.set_xticks(x_base)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel('Validation Accuracy')
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.legend(fontsize=8, loc='lower right', ncol=2)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_FILE}")
    plt.show()


if __name__ == '__main__':
    plot()
