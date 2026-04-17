"""
Convenience runner for the full pipeline.

Usage:
    python run_pipeline.py [--phase PHASE]

Phases:
    1  extract   — extract DINO features  (run once, needs internet)
    2  annotate  — start the Flask annotation tool (collect human labels)
    3  sweep     — run the synthetic noise sweep  (slow, ~hours)
    4  real       — run the real data experiment
    5  figures   — generate all 7 figures from saved results

Examples:
    python run_pipeline.py --phase extract
    python run_pipeline.py --phase annotate
    python run_pipeline.py --phase sweep
    python run_pipeline.py --phase figures
"""

import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent   # mylibs/


def phase_extract():
    print("=== Phase 1: Feature Extraction ===")
    subprocess.run([sys.executable, str(ROOT / 'extract_features.py')], check=True)


def phase_annotate():
    print("=== Phase 2 (Human): Annotation Tool ===")
    print("Open http://localhost:5000 in your browser.")
    subprocess.run([sys.executable, str(ROOT / 'annotations' / 'annotationTool.py')],
                   check=True)


def phase_sweep():
    print("=== Phase 4A: Synthetic Sweep ===")
    subprocess.run([sys.executable, '-m', 'experiments.synthetic_sweep'],
                   cwd=str(ROOT), check=True)


def phase_real():
    print("=== Phase 4B: Real Data Experiment ===")
    subprocess.run([sys.executable, '-m', 'experiments.real_data'],
                   cwd=str(ROOT), check=True)


def phase_figures():
    print("=== Phase 5: Figures ===")
    scripts = [
        'visualize.accuracy_curves',
        'visualize.loss_histogram',
        'visualize.transition_matrix',
        'visualize.flagged_samples',
        'visualize.calibration_plot',
        'visualize.learning_curves',
        'visualize.real_data_bar',
    ]
    for mod in scripts:
        print(f"\n--- {mod} ---")
        subprocess.run([sys.executable, '-m', mod], cwd=str(ROOT), check=True)


PHASES = {
    'extract':  phase_extract,
    'annotate': phase_annotate,
    'sweep':    phase_sweep,
    'real':     phase_real,
    'figures':  phase_figures,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=list(PHASES.keys()), required=True)
    args = parser.parse_args()
    PHASES[args.phase]()
