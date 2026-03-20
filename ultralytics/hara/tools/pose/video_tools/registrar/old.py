

import os
import cv2
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import point_visualization


"""
two cameras:
- have different viewpoints
- have different scale / offset
- have different distortion
- and are also cross-modal (RGB/visual vs thermal)
"""


# ============================================================
# 1. Reproducibility
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Dataset
# ============================================================
class PointMappingDataset(Dataset):
    """
    CSV format:
        x1,y1,x2,y2
        ...
    """

    def __init__(
            self,
            csv_path: str,
            img1_size: Tuple[int, int],
            img2_size: Tuple[int, int],
    ):
        """
        img1_size: (width1, height1)
        img2_size: (width2, height2)
        """
        self.df = pd.read_csv(csv_path)

        required_cols = ["x1", "y1", "x2", "y2"]
        for c in required_cols:
            if c not in self.df.columns:
                raise ValueError(f"Missing column '{c}' in CSV.")

        self.w1, self.h1 = img1_size
        self.w2, self.h2 = img2_size

        self.inputs = self.df[["x1", "y1"]].values.astype(np.float32)
        self.targets = self.df[["x2", "y2"]].values.astype(np.float32)

        # Normalize coordinates to [0, 1]
        self.inputs_norm = self.inputs.copy()
        self.inputs_norm[:, 0] /= self.w1
        self.inputs_norm[:, 1] /= self.h1

        self.targets_norm = self.targets.copy()
        self.targets_norm[:, 0] /= self.w2
        self.targets_norm[:, 1] /= self.h2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs_norm[idx], dtype=torch.float32)
        y = torch.tensor(self.targets_norm[idx], dtype=torch.float32)
        return x, y


# ============================================================
# 3. Model
# ============================================================
class CoordinateMapperNN(nn.Module):
    """
    Simple MLP for learning nonlinear coordinate transform.
    Input : (x1, y1)
    Output: (x2, y2)
    """

    def __init__(self, hidden_dim: int = 128, depth: int = 4, dropout: float = 0.1):
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
# 4. Training and evaluation utilities
# ============================================================
def compute_rmse_pixels(
        preds_norm: torch.Tensor,
        targets_norm: torch.Tensor,
        img2_size: Tuple[int, int]
) -> float:
    """
    Convert normalized prediction/target back to pixel coordinates
    and compute RMSE in pixel space.
    """
    w2, h2 = img2_size

    preds_px = preds_norm.clone()
    targets_px = targets_norm.clone()

    preds_px[:, 0] *= w2
    preds_px[:, 1] *= h2

    targets_px[:, 0] *= w2
    targets_px[:, 1] *= h2

    mse = torch.mean((preds_px - targets_px) ** 2)
    rmse = torch.sqrt(mse).item()
    return rmse


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
def evaluate(model, loader, criterion, device, img2_size):
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

        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    rmse_px = compute_rmse_pixels(all_preds, all_targets, img2_size)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, rmse_px, all_preds, all_targets


# ============================================================
# 5. Prediction helpers
# ============================================================
@torch.no_grad()
def transform_points(
        model: nn.Module,
        points_img1: np.ndarray,
        img1_size: Tuple[int, int],
        img2_size: Tuple[int, int],
        device: str = "cpu"
) -> np.ndarray:
    """
    points_img1: shape (N, 2), pixel coordinates in image1
    returns: shape (N, 2), predicted pixel coordinates in image2
    """
    if len(points_img1.shape) != 2 or points_img1.shape[1] != 2:
        raise ValueError("points_img1 must have shape (N, 2)")

    w1, h1 = img1_size
    w2, h2 = img2_size

    pts = points_img1.astype(np.float32).copy()
    pts[:, 0] /= w1
    pts[:, 1] /= h1

    x = torch.tensor(pts, dtype=torch.float32).to(device)

    model.eval()
    pred_norm = model(x).cpu().numpy()

    pred_px = pred_norm.copy()
    pred_px[:, 0] *= w2
    pred_px[:, 1] *= h2

    return pred_px


def visualize_dense_grid_mapping(
        model: nn.Module,
        img1: np.ndarray,
        img2: np.ndarray,
        img1_size: Tuple[int, int],
        img2_size: Tuple[int, int],
        grid_step: int = 100,
        output_path: str = "grid_mapping.png",
        device: str = "cpu"
):
    """
    Create a visual comparison using a regular grid sampled from image1,
    then map each point to image2.
    """
    h1, w1 = img1.shape[:2]

    xs = np.arange(grid_step, w1, grid_step)
    ys = np.arange(grid_step, h1, grid_step)

    grid_points = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
    pred_points = transform_points(model, grid_points, img1_size, img2_size, device=device)

    point_visualization.draw_point_pairs(
        img1=img1,
        img2=img2,
        src_points=grid_points,
        pred_points_img2=pred_points,
        gt_points_img2=None,
        output_path=output_path,
        label_prefix="g"
    )


# ============================================================
# 7. Main training pipeline
# ============================================================
def main():
    set_seed(42)

    # --------------------------------------------------------
    # User settings
    # --------------------------------------------------------
    csv_path = r"/ultralytics/hara/tools/pose/video_tools/.captured_frames/registrar/readme.csv"
    image1_path = r"/ultralytics/hara/tools/pose/video_tools/.captured_frames/registrar/rgb_frame_00_00_01.png"
    image2_path = r"/ultralytics/hara/tools/pose/video_tools/.captured_frames/registrar/thermal_00_00_01.png"

    batch_size = 32
    num_epochs = 500
    learning_rate = 1e-3
    hidden_dim = 128
    depth = 4
    dropout = 0.05
    train_ratio = 0.8

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None:
        raise FileNotFoundError(f"Cannot read image1: {image1_path}")
    if img2 is None:
        raise FileNotFoundError(f"Cannot read image2: {image2_path}")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    img1_size = (w1, h1)
    img2_size = (w2, h2)

    print(f"Image1 size: {img1_size}")
    print(f"Image2 size: {img2_size}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    dataset = PointMappingDataset(csv_path, img1_size, img2_size)

    n_total = len(dataset)
    n_train = max(1, int(n_total * train_ratio))
    n_test = n_total - n_train
    if n_test == 0:
        raise ValueError("Need at least 2 samples so that test set is not empty.")

    train_set, test_set = random_split(
        dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CoordinateMapperNN(hidden_dim=hidden_dim, depth=depth, dropout=dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    best_test_loss = float("inf")
    best_model_path = "../best_mapper_model.pth"

    train_losses = []
    test_losses = []
    test_rmses = []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_rmse_px, _, _ = evaluate(model, test_loader, criterion, device, img2_size)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_rmses.append(test_rmse_px)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), best_model_path)

        if epoch % 50 == 0 or epoch == 1 or epoch == num_epochs:
            print(
                f"Epoch [{epoch:4d}/{num_epochs}] | "
                f"Train Loss: {train_loss:.6f} | "
                f"Test Loss: {test_loss:.6f} | "
                f"Test RMSE(px): {test_rmse_px:.3f}"
            )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model from: {best_model_path}")

    # --------------------------------------------------------
    # Final evaluation on test set
    # --------------------------------------------------------
    test_loss, test_rmse_px, all_preds_norm, all_targets_norm = evaluate(
        model, test_loader, criterion, device, img2_size
    )
    print(f"\nFinal Test Loss: {test_loss:.6f}")
    print(f"Final Test RMSE(px): {test_rmse_px:.3f}")

    # --------------------------------------------------------
    # Plot training curves
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=200)
    plt.close()
    print("Saved: loss_curve.png")

    plt.figure(figsize=(8, 5))
    plt.plot(test_rmses)
    plt.xlabel("Epoch")
    plt.ylabel("Test RMSE (pixels)")
    plt.title("Test RMSE")
    plt.tight_layout()
    plt.savefig("rmse_curve.png", dpi=200)
    plt.close()
    print("Saved: rmse_curve.png")

    # --------------------------------------------------------
    # Visualize test correspondences
    # --------------------------------------------------------
    # Recover exact test samples from subset indices
    test_indices = test_set.indices
    df = pd.read_csv(csv_path)

    src_points_test = df.loc[test_indices, ["x1", "y1"]].values.astype(np.float32)
    gt_points_test = df.loc[test_indices, ["x2", "y2"]].values.astype(np.float32)

    pred_points_test = transform_points(
        model,
        src_points_test,
        img1_size=img1_size,
        img2_size=img2_size,
        device=device
    )

    point_visualization.draw_point_pairs(
        img1=img1,
        img2=img2,
        src_points=src_points_test,
        pred_points_img2=pred_points_test,
        gt_points_img2=gt_points_test,
        output_path="../test_point_comparison.png",
        label_prefix="t"
    )

    # --------------------------------------------------------
    # Visualize dense grid mapping
    # --------------------------------------------------------
    visualize_dense_grid_mapping(
        model=model,
        img1=img1,
        img2=img2,
        img1_size=img1_size,
        img2_size=img2_size,
        grid_step=120,
        output_path="../dense_grid_mapping.png",
        device=device
    )

    # --------------------------------------------------------
    # Example: transform some custom points
    # --------------------------------------------------------
    custom_points_img1 = np.array([
        [300, 300],
        [600, 500],
        [900, 700],
    ], dtype=np.float32)

    custom_pred_img2 = transform_points(
        model,
        custom_points_img1,
        img1_size=img1_size,
        img2_size=img2_size,
        device=device
    )

    print("\nCustom point mapping:")
    for src, dst in zip(custom_points_img1, custom_pred_img2):
        print(f"Image1 {src}  -->  Image2 {dst}")


if __name__ == "__main__":
    main()