import torch
import torch.nn as nn
from torchvision import models


def get_classifier(num_classes=2, backbone='resnet50', pretrained=True):
    """Standard transfer learning setup: pretrained backbone + classification head"""
    if backbone == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif backbone == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif backbone == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif backbone == 'mobilenet_v3':
        model = models.mobilenet_v3_small(pretrained=pretrained)
        # Rebuild classifier: features(576) -> Linear(1024) -> Hardswish -> Dropout -> Linear(num_classes)
        model.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes)
        )
    else:
        raise ValueError(f'Unsupported backbone: {backbone}')
    
    # freeze early layers, unfreeze last few children (same as proto)
    children = list(model.named_children())
    n = len(children)
    to_unfreeze = set(name for name, _ in children[max(0, n - 2):])
    for name, child in children:
        requires = name in to_unfreeze
        for p in child.parameters():
            p.requires_grad = requires
            
    return model