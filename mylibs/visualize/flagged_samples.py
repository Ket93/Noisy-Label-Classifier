# figure 4

import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent   # mylibs/
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dataset import load_data, CLASSES
from noise.factory  import apply_noise

AUX_FILE    = ROOT / 'results' / 'synthetic_aux.json'
OUTPUT_FILE = ROOT / 'results' / 'fig4_flagged_samples.png'

GRID_ROWS = 4
GRID_COLS = 6
MAX_SHOW  = GRID_ROWS * GRID_COLS


def plot():
    with open(AUX_FILE) as f:
        aux = json.load(f)

    flagged_global = aux.get('cl_flagged')
    if flagged_global is None:
        print("No flagged indices in synthetic_aux.json. Run the sweep first.")
        return

    features, labels, image_paths, train_idx, val_idx = load_data(ROOT)
    train_features = features[train_idx]
    train_labels   = labels[train_idx]

    # Reconstruct 30% uniform noise (seed=0) to know what labels were given
    noisy_train = apply_noise(train_features, train_labels, 'uniform', 0.3, seed=0)

    # flagged_global are indices into the TRAINING array (from CL's perspective)
    flagged = np.array(flagged_global, dtype=np.int64)
    flagged = flagged[flagged < len(train_idx)]  # safety

    # Build suggested labels: argmax of CL OOF probs
    # (We'll use noisy label vs true label as "given" vs "suggested")
    show_idx = flagged[:MAX_SHOW]
    n_show   = len(show_idx)

    if n_show == 0:
        print("No flagged samples to display.")
        return

    rows = (n_show + GRID_COLS - 1) // GRID_COLS
    fig, axes = plt.subplots(rows, GRID_COLS, figsize=(GRID_COLS * 2.2, rows * 2.6))
    fig.suptitle('Samples Flagged by Confident Learning as Mislabeled\n'
                 '(30% Uniform Noise)', fontsize=11)

    axes_flat = np.array(axes).ravel()

    for i, train_pos in enumerate(show_idx):
        global_idx  = train_idx[train_pos]
        img_path    = str(image_paths[global_idx])
        given_lbl   = CLASSES[noisy_train[train_pos]]
        true_lbl    = CLASSES[train_labels[train_pos]]

        ax = axes_flat[i]
        try:
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center')

        color = 'red' if given_lbl != true_lbl else 'green'
        ax.set_title(f'Given: {given_lbl}\nTrue: {true_lbl}',
                     fontsize=7, color=color)
        ax.axis('off')

    # Hide unused axes
    for j in range(n_show, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=120, bbox_inches='tight')
    print(f"Saved: {OUTPUT_FILE}")
    plt.show()


if __name__ == '__main__':
    plot()
