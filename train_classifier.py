import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from models.classifier import get_classifier
from data_loader import default_transform
from utils import save_checkpoint, compute_metrics, plot_confusion, set_logger


class ImageFolderDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or default_transform()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def train_classifier(cfg):
    """Train a standard classifier (no prototypical/few-shot learning)"""
    set_logger(cfg.get('log_file', None))
    
    # device setup
    dev = cfg.get('device', 'cpu')
    if dev == 'mps' and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif dev == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    # prepare data (80-20 split like proto)
    data_root = cfg['data_root']
    train_imgs, train_labels = [], []
    val_imgs, val_labels = [], []
    for label, cls in enumerate(['normal', 'infected']):
        cls_path = os.path.join(data_root, cls)
        if not os.path.isdir(cls_path):
            continue
        imgs = []
        for fn in os.listdir(cls_path):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                imgs.append(os.path.join(cls_path, fn))
        # 80-20 split
        split = int(0.8 * len(imgs))
        train_imgs.extend(imgs[:split])
        train_labels.extend([label] * split)
        val_imgs.extend(imgs[split:])
        val_labels.extend([label] * (len(imgs) - split))
    
    train_ds = ImageFolderDataset(train_imgs, train_labels)
    val_ds = ImageFolderDataset(val_imgs, val_labels)
    train_loader = DataLoader(train_ds, batch_size=cfg.get('batch_size', 16), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get('batch_size', 16))
    
    # model & training setup
    model = get_classifier(num_classes=2, backbone=cfg['backbone']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=float(cfg.get('lr', 1e-4)))
    
    best_val_acc = 0
    epochs = int(cfg.get('epochs', 50))  # roughly equivalent time to proto episodes
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # validation
        model.eval()
        val_loss = 0
        val_preds, val_labels_all = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels_all.extend(labels.cpu().numpy())
                
        val_metrics = compute_metrics(val_labels_all, val_preds)
        val_acc = val_metrics['accuracy']
        
        logging.info(f'Epoch {epoch+1}/{epochs} train_loss={train_loss/len(train_loader):.4f} '
                    f'val_loss={val_loss/len(val_loader):.4f} val_acc={val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'val_acc': val_acc,
                'cfg': cfg,
            }, os.path.join(cfg.get('checkpoint_dir', 'checkpoints'), 'classifier_best.pth'))
    
    # final save
    save_checkpoint({
        'model_state': model.state_dict(),
        'val_acc': val_acc,
        'cfg': cfg,
    }, os.path.join(cfg.get('checkpoint_dir', 'checkpoints'), 'classifier_final.pth'))
    logging.info('Training finished')