import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import point_visualization
import NN_fit
import point_utility
import TPS_fit

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





def visualize_dense_grid_mapping(
        transform_fn,
        img1: np.ndarray,
        img2: np.ndarray,
        img1_size: Tuple[int, int],
        img2_size: Tuple[int, int],
        grid_step: int = 100,
        output_path: str = "grid_mapping.png",
):
    h1, w1 = img1.shape[:2]
    xs = np.arange(grid_step, w1, grid_step)
    ys = np.arange(grid_step, h1, grid_step)

    grid_points = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
    pred_points = transform_fn(grid_points)

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

    # method="tps"
    method="nn"
    # --------------------------------------------------------
    # User settings
    # --------------------------------------------------------
    csv_path = r"/ultralytics/hara/tools/pose/video_tools/registrar/.data/readme.csv"
    image1_path = r"/ultralytics/hara/tools/pose/video_tools/registrar/.data/rgb_frame_00_00_01.png"
    image2_path = r"/ultralytics/hara/tools/pose/video_tools/registrar/.data/thermal_00_00_01.png"

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

    dataset = PointMappingDataset(csv_path, img1_size, img2_size)

    n_total = len(dataset)
    n_train = max(3, int(n_total * 0.8))
    n_test = n_total - n_train
    if n_test <= 0:
        raise ValueError("Need enough points so test set is not empty.")
    if n_train < 3:
        raise ValueError("TPS needs at least 3 training points.")

    train_set, test_set = random_split(
        dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    df = pd.read_csv(csv_path)
    train_indices = train_set.indices
    test_indices = test_set.indices

    src_train_px = df.loc[train_indices, ["x1", "y1"]].values.astype(np.float32)
    dst_train_px = df.loc[train_indices, ["x2", "y2"]].values.astype(np.float32)

    src_test_px = df.loc[test_indices, ["x1", "y1"]].values.astype(np.float32)
    dst_test_px = df.loc[test_indices, ["x2", "y2"]].values.astype(np.float32)

    if method == "nn":
        batch_size = 32
        num_epochs = 500
        learning_rate = 1e-3
        hidden_dim = 128
        depth = 4
        dropout = 0.05

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = NN_fit.CoordinateMapperNN(hidden_dim=hidden_dim, depth=depth, dropout=dropout).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_test_loss = float("inf")
        best_model_path = "best_mapper_model.pth"

        train_losses = []
        test_losses = []
        test_rmses = []

        for epoch in range(1, num_epochs + 1):
            train_loss = NN_fit.train_one_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_rmse_px, _, _ = NN_fit.evaluate_nn(model, test_loader, criterion, device, img2_size)

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

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from: {best_model_path}")

        test_loss, test_rmse_px, _, _ = NN_fit.evaluate_nn(model, test_loader, criterion, device, img2_size)
        print(f"\n[NN] Final Test Loss: {test_loss:.6f}")
        print(f"[NN] Final Test RMSE(px): {test_rmse_px:.3f}")

        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("NN Training Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("loss_curve_nn.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(test_rmses)
        plt.xlabel("Epoch")
        plt.ylabel("Test RMSE (pixels)")
        plt.title("NN Test RMSE")
        plt.tight_layout()
        plt.savefig("rmse_curve_nn.png", dpi=200)
        plt.close()

        pred_test_px = NN_fit.transform_points_nn(model, src_test_px, img1_size, img2_size, device=device)

        point_visualization.draw_point_pairs(
            img1=img1,
            img2=img2,
            src_points=src_test_px,
            pred_points_img2=pred_test_px,
            gt_points_img2=dst_test_px,
            output_path="test_point_comparison_nn.png",
            label_prefix="t"
        )

        visualize_dense_grid_mapping(
            transform_fn=lambda pts: NN_fit.transform_points_nn(model, pts, img1_size, img2_size, device=device),
            img1=img1,
            img2=img2,
            img1_size=img1_size,
            img2_size=img2_size,
            grid_step=120,
            output_path="dense_grid_mapping_nn.png"
        )

    elif method == "tps":
        src_train_norm = point_utility.normalize_points(src_train_px, img1_size)
        dst_train_norm = point_utility.normalize_points(dst_train_px, img2_size)

        tps = TPS_fit.TPSMapper(reg=1e-6)
        tps.fit(src_train_norm, dst_train_norm)

        test_loss, test_rmse_px, _, _ = TPS_fit.evaluate_tps(
            tps,
            src_test_px,
            dst_test_px,
            img1_size,
            img2_size
        )

        print(f"\n[TPS] Test Loss (normalized MSE): {test_loss:.6f}")
        print(f"[TPS] Test RMSE(px): {test_rmse_px:.3f}")

        pred_test_px = TPS_fit.transform_points_tps(tps, src_test_px, img1_size, img2_size)

        point_visualization.draw_point_pairs(
            img1=img1,
            img2=img2,
            src_points=src_test_px,
            pred_points_img2=pred_test_px,
            gt_points_img2=dst_test_px,
            output_path="test_point_comparison_tps.png",
            label_prefix="t"
        )

        visualize_dense_grid_mapping(
            transform_fn=lambda pts: TPS_fit.transform_points_tps(tps, pts, img1_size, img2_size),
            img1=img1,
            img2=img2,
            img1_size=img1_size,
            img2_size=img2_size,
            grid_step=120,
            output_path="dense_grid_mapping_tps.png"
        )

        custom_points_img1 = np.array([
            [300, 300],
            [600, 500],
            [900, 700],
        ], dtype=np.float32)

        custom_pred_img2 = TPS_fit.transform_points_tps(tps, custom_points_img1, img1_size, img2_size)

        print("\n[TPS] Custom point mapping:")
        for src, dst in zip(custom_points_img1, custom_pred_img2):
            print(f"Image1 {src} --> Image2 {dst}")

    else:
        raise ValueError("method must be either 'nn' or 'tps'")


if __name__ == "__main__":
    main()