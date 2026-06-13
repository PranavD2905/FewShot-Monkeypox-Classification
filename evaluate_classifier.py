import os
import torch
from tqdm import trange
import numpy as np
from torch.utils.data import DataLoader

from models.classifier import get_classifier
from train_classifier import ImageFolderDataset
from utils import compute_metrics, plot_confusion


def evaluate_classifier(cfg, checkpoint_path=None):
    """Evaluate standard classifier (no prototypical/few-shot learning)"""
    # device setup
    dev = cfg.get('device', 'cpu')
    if dev == 'mps' and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif dev == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    # load test data (use validation split like proto for fair comparison)
    data_root = cfg['data_root']
    test_imgs, test_labels = [], []
    for label, cls in enumerate(['normal', 'infected']):
        cls_path = os.path.join(data_root, cls)
        if not os.path.isdir(cls_path):
            continue
        imgs = []
        for fn in os.listdir(cls_path):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                imgs.append(os.path.join(cls_path, fn))
        # use last 20% like proto eval
        split = int(0.8 * len(imgs))
        test_imgs.extend(imgs[split:])
        test_labels.extend([label] * (len(imgs) - split))
    
    test_ds = ImageFolderDataset(test_imgs, test_labels)
    test_loader = DataLoader(test_ds, batch_size=cfg.get('batch_size', 16))
    
    # load model
    model = get_classifier(num_classes=2, backbone=cfg['backbone']).to(device)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # prob of class 1 (infected)
            
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    print('Evaluation results:', metrics)
    
    # confusion matrix
    plot_confusion(all_labels, all_preds, ['normal', 'infected'],
                  os.path.join(cfg.get('output_dir', 'outs'), 'confusion_classifier.png'))
    
    return metrics