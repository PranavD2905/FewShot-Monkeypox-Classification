import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)


def save_config(cfg, path):
    with open(path, 'w') as f:
        yaml.dump(cfg, f)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(state, path):
    torch.save(state, path)


def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device)


def compute_metrics(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary')
    rec = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    auc = None
    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = None
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, auc=auc)


def plot_curves(logs: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    if 'loss' in logs:
        plt.plot(logs['loss'], label='loss')
    if 'val_loss' in logs:
        plt.plot(logs['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def plot_confusion(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred)
    
    # For binary classification, extract TP, TN, FP, FN
    if len(classes) == 2:
        tn, fp, fn, tp = cm.ravel()
    
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    
    # Add text annotations with counts and percentages
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = count / cm.sum() * 100
            
            # Determine label
            if len(classes) == 2:
                if i == 0 and j == 0:
                    label_type = f'TN\n(True Negative)'
                elif i == 0 and j == 1:
                    label_type = f'FP\n(False Positive)'
                elif i == 1 and j == 0:
                    label_type = f'FN\n(False Negative)'
                else:  # i == 1 and j == 1
                    label_type = f'TP\n(True Positive)'
            else:
                label_type = ''
            
            text = f'{label_type}\n{count}\n({percentage:.1f}%)'
            plt.text(j, i, text,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11, fontweight='bold')
    
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    # Add metrics summary text box
    if len(classes) == 2:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics_text = (
            f'Metrics:\n'
            f'Accuracy: {accuracy:.3f}\n'
            f'Precision: {precision:.3f}\n'
            f'Recall: {recall:.3f}\n'
            f'Specificity: {specificity:.3f}'
        )
        plt.text(1.35, 0.5, metrics_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def set_logger(log_file=None, level=logging.INFO):
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, handlers=handlers,
                        format='%(asctime)s - %(levelname)s - %(message)s')
