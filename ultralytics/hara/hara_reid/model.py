import torch
import torch.nn as nn


def build_osnet(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build OSNet using torchreid.

    In training mode with loss='triplet', torchreid OSNet returns (logits, features).
    In eval mode, it returns features only.
    """
    try:
        from torchreid import models
    except Exception as e:
        raise ImportError(
            "Cannot import torchreid. Try: pip install torchreid\n"
            "If that fails: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
        ) from e

    model = models.build_model(
        name=model_name,
        num_classes=num_classes,
        loss="triplet",
        pretrained=pretrained,
    )
    return model


class BatchHardTripletLoss(nn.Module):
    """Batch-hard triplet loss for ReID embeddings."""
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Pairwise Euclidean distance
        dist = torch.cdist(embeddings, embeddings, p=2)
        labels = labels.view(-1, 1)
        mask_pos = labels.eq(labels.t())
        mask_neg = ~mask_pos
        mask_pos.fill_diagonal_(False)

        if mask_pos.sum() == 0 or mask_neg.sum() == 0:
            return embeddings.new_tensor(0.0, requires_grad=True)

        # hardest positive: max distance among same-ID images
        dist_ap = torch.where(mask_pos, dist, torch.zeros_like(dist)).max(dim=1)[0]
        # hardest negative: min distance among different-ID images
        large = torch.full_like(dist, 1e6)
        dist_an = torch.where(mask_neg, dist, large).min(dim=1)[0]

        valid = (dist_ap > 0) & (dist_an < 1e6)
        if valid.sum() == 0:
            return embeddings.new_tensor(0.0, requires_grad=True)

        y = torch.ones_like(dist_an[valid])
        return self.ranking_loss(dist_an[valid], dist_ap[valid], y)
