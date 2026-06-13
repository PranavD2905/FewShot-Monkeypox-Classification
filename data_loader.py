import os
import random
from typing import List, Tuple, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FolderFewShotDataset(Dataset):
    """
    Simple folder-based dataset that lists images per class and returns filepaths.
    Expects directory structure:
        root/class_x/*.jpg
        root/class_y/*.png

    We'll not use Dataset.__getitem__ for episodic sampling; instead we expose class_to_images.
    """

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.class_to_images = self._scan()
        self.classes = sorted(list(self.class_to_images.keys()))

    def _scan(self) -> Dict[str, List[str]]:
        d = {}
        for cls in sorted(os.listdir(self.root)):
            cls_path = os.path.join(self.root, cls)
            if not os.path.isdir(cls_path):
                continue
            imgs = []
            for fn in os.listdir(cls_path):
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    imgs.append(os.path.join(cls_path, fn))
            if imgs:
                d[cls] = imgs
        return d

    def split_train_val(self, val_ratio=0.2, seed=42):
        """Split each class into train/val lists and return dicts.
        Returns: train_dict, val_dict mapping class->list(paths)
        """
        random.seed(seed)
        train = {}
        val = {}
        for cls, imgs in self.class_to_images.items():
            imgs_copy = imgs.copy()
            random.shuffle(imgs_copy)
            n_val = max(1, int(len(imgs_copy) * val_ratio))
            val[cls] = imgs_copy[:n_val]
            train[cls] = imgs_copy[n_val:]
        return train, val


def default_transform(img_size=224):
    """Legacy default eval transform (no heavy augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def default_transform_train(img_size=224):
    """Train-time augmentations (kept conservative for medical imaging)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def default_transform_eval(img_size=224):
    """Eval-time transform (no augmentation)."""
    return default_transform(img_size)


def make_episode(class_to_images: Dict[str, List[str]], N: int, K: int, Q: int, transform, device):
    """
    Sample an episode: return tensors support_images, support_labels, query_images, query_labels
    support_images: (N*K, C, H, W)  support_labels: (N*K,) in 0..N-1
    query_images: (N*Q, C, H, W)    query_labels: (N*Q,)
    """
    chosen_classes = random.sample(list(class_to_images.keys()), N)
    support_images, support_labels = [], []
    query_images, query_labels = [], []

    for cls_idx, cls in enumerate(chosen_classes):
        imgs = class_to_images[cls]
        if len(imgs) < K + Q:
            # allow sampling with replacement if insufficient
            samples = random.choices(imgs, k=K + Q)
        else:
            samples = random.sample(imgs, K + Q)
        support_paths = samples[:K]
        query_paths = samples[K:K + Q]

        for p in support_paths:
            im = Image.open(p).convert('RGB')
            support_images.append(transform(im))
            support_labels.append(cls_idx)

        for p in query_paths:
            im = Image.open(p).convert('RGB')
            query_images.append(transform(im))
            query_labels.append(cls_idx)

    support_images = torch.stack(support_images).to(device)
    support_labels = torch.tensor(support_labels, dtype=torch.long, device=device)
    query_images = torch.stack(query_images).to(device)
    query_labels = torch.tensor(query_labels, dtype=torch.long, device=device)

    return support_images, support_labels, query_images, query_labels


if __name__ == '__main__':
    # quick debug
    ds = FolderFewShotDataset('data', transform=default_transform())
    print('Classes:', ds.classes)