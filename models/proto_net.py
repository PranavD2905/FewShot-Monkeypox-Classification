import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network wrapper that uses a pretrained backbone to compute embeddings.
    Follows Snell et al., 2017: prototypes = mean of support embeddings per class.
    Prediction is done with softmax over negative Euclidean distances.
    """

    def __init__(self, backbone: nn.Module, distance: str = 'euclidean', temperature: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.distance = distance.lower()
        self.temperature = float(temperature)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        """
        support_x: (N*K, C, H, W)
        support_y: (N*K,)
        query_x: (N*Q, C, H, W)

        Returns:
          log_p_y: (N*Q, N) log-probabilities for each class
        """
        emb_support = self.embed(support_x)  # (N*K, D)
        emb_query = self.embed(query_x)      # (N*Q, D)

        classes = torch.unique(support_y)
        classes = classes.sort()[0]
        prototypes = []
        for c in classes:
            inds = (support_y == c).nonzero(as_tuple=True)[0]
            proto = emb_support[inds].mean(dim=0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes)  # (N, D)

        if self.distance == 'cosine':
            # embeddings are already L2-normalized in backbone; ensure prototypes are too
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            # cosine similarity: (M, D) @ (D, N) -> (M, N)
            sims = emb_query @ proto_norm.t()
            logits = sims / self.temperature
        else:
            # squared Euclidean distance
            diff = emb_query.unsqueeze(1) - prototypes.unsqueeze(0)
            dists = (diff ** 2).sum(dim=2)
            logits = (-dists) / self.temperature

        log_p_y = F.log_softmax(logits, dim=1)
        return log_p_y


if __name__ == '__main__':
    # small smoke test
    class DummyBackbone(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.d = d

        def forward(self, x):
            batch = x.shape[0]
            return torch.randn(batch, self.d)

    model = PrototypicalNetwork(DummyBackbone())
    s_x = torch.randn(10, 3, 224, 224)
    s_y = torch.tensor([0,0,0,0,0,1,1,1,1,1])
    q_x = torch.randn(6, 3, 224, 224)
    out = model(s_x, s_y, q_x)
    print(out.shape)