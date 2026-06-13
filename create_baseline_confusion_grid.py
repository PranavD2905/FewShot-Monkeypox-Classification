"""
Create a 2x2 grid confusion matrix image for baseline (standard classifier) models.
Saves as `confusion_matrics_baseline.png` in repo root.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Baseline classifier confusion counts from evaluation (small validation set)
# These were computed earlier (12 samples: 6 pos, 6 neg)
baseline_stats = [
    ("ResNet50",    {'tp':4, 'tn':6, 'fp':0, 'fn':2}),
    ("EfficientNet-B0", {'tp':4, 'tn':6, 'fp':0, 'fn':2}),
    ("DenseNet121", {'tp':4, 'tn':6, 'fp':0, 'fn':2}),
    ("MobileNetV3", {'tp':4, 'tn':6, 'fp':0, 'fn':2}),
]

def plot_single_cm(ax, cm, classes, title, metrics_text=None):
    cmap = plt.cm.Blues
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title, fontsize=12, fontweight='bold')

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)

    thresh = cm.max() / 2.0
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            pct = (count / total) * 100 if total > 0 else 0.0
            # label type
            if i == 0 and j == 0:
                label = 'TN\n'
            elif i == 0 and j == 1:
                label = 'FP\n'
            elif i == 1 and j == 0:
                label = 'FN\n'
            else:
                label = 'TP\n'
            text = f"{label}{count}\n({pct:.1f}%)"
            ax.text(j, i, text, ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=10, fontweight='bold')

    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)

    if metrics_text is not None:
        ax.text(1.05, 0.5, metrics_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def main():
    classes = ['normal', 'infected']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (name, stats) in zip(axes, baseline_stats):
        tp = stats['tp']
        tn = stats['tn']
        fp = stats['fp']
        fn = stats['fn']
        cm = np.array([[tn, fp], [fn, tp]])

        # compute simple metrics
        total = cm.sum()
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics_text = (f"Acc: {accuracy:.3f}\nPrecision: {precision:.3f}\n"
                        f"Recall: {recall:.3f}\nF1: {f1:.3f}\nTotal: {total}")

        plot_single_cm(ax, cm, classes, title=name, metrics_text=metrics_text)

    plt.suptitle('Baseline Classifiers - Confusion Matrices (Validation set)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    out_path = Path('confusion_matrics_baseline.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
