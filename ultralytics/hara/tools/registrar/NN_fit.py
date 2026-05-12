import numpy as np

from typing import Tuple

import torch
import torch.nn as nn

import point_utility



# ============================================================
# 3. Neural Network Mapper
# ============================================================
class CoordinateMapperNN(nn.Module):
    def __init__(self, hidden_dim: int = 128, depth: int = 4, dropout: float = 0.05):
        super().__init__()

        layers = []
        in_dim = 2
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)




# ============================================================
# 6. NN training and evaluation
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_nn(model, loader, criterion, device, img2_size):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)

        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    pred_px = point_utility.denormalize_points(all_preds, img2_size)
    gt_px = point_utility.denormalize_points(all_targets, img2_size)
    rmse_px = point_utility.compute_rmse_pixels_from_arrays(pred_px, gt_px)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, rmse_px, all_preds, all_targets


@torch.no_grad()
def transform_points_nn(
        model: nn.Module,
        points_img1: np.ndarray,
        img1_size: Tuple[int, int],
        img2_size: Tuple[int, int],
        device: str = "cpu"
) -> np.ndarray:
    pts_norm = point_utility.normalize_points(points_img1, img1_size)
    x = torch.tensor(pts_norm, dtype=torch.float32).to(device)

    model.eval()
    pred_norm = model(x).cpu().numpy()
    pred_px = point_utility.denormalize_points(pred_norm, img2_size)
    return pred_px
