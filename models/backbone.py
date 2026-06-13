import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


def _freeze_children(module: nn.Module, unfrozen_children: int = 2):
    """Freeze all children except the last `unfrozen_children` named children.
    This is a heuristic that works across ResNet, DenseNet, EfficientNet, MobileNet.
    """
    children = list(module.named_children())
    n = len(children)
    to_unfreeze = set(name for name, _ in children[max(0, n - unfrozen_children):])
    for name, child in children:
        requires = name in to_unfreeze
        for p in child.parameters():
            p.requires_grad = requires


def get_backbone(name: str = 'resnet50', embedding_dim: int = 512, fine_tune_layers: int = 2, pretrained: bool = True, dropout_p: float = 0.0) -> nn.Module:
    name = name.lower()
    if name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        backbone = model
    elif name == 'densenet121' or name == 'densenet':
        model = models.densenet121(pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Identity()
        backbone = model
    elif name == 'efficientnet_b0' or name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
        backbone = model
    elif name == 'mobilenet_v3' or name == 'mobilenetv3':
        model = models.mobilenet_v3_small(pretrained=pretrained)
        # MobileNetV3 has features->avgpool->flatten->classifier
        # classifier[0] is Linear(576, 1024), so we need features output = 576
        # but after avgpool it's 576, before flatten it's (B, 576, 7, 7)
        # We'll use 576 as in_features and ensure we pool/flatten in forward
        in_features = 576
        model.classifier = nn.Identity()
        backbone = model
    else:
        raise ValueError(f'Unsupported backbone: {name}')

    # projection head to embedding_dim
    # optional dropout before projection for regularization
    if dropout_p and dropout_p > 0.0:
        projection = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, embedding_dim))
    else:
        projection = nn.Linear(in_features, embedding_dim)

    class BackboneWithProjection(nn.Module):
        def __init__(self, net, proj):
            super().__init__()
            self.net = net
            self.proj = proj

        def forward(self, x):
            x = self.net(x)
            if x.ndim == 4:
                # some nets may return feature maps; global pool
                x = torch.flatten(x, 1)
            x = self.proj(x)
            # if projection is a Sequential with dropout+Linear, output is still (B, D)
            x = nn.functional.normalize(x, p=2, dim=1)  # L2 normalize embeddings
            return x

    model = BackboneWithProjection(backbone, projection)

    # freeze early layers, unfreeze last few children
    _freeze_children(backbone, unfrozen_children=fine_tune_layers)

    return model


if __name__ == '__main__':
    m = get_backbone('resnet50', embedding_dim=512, fine_tune_layers=2)
    print(m)