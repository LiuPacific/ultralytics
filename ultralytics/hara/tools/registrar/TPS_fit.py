import numpy as np

from typing import Tuple

import point_utility

# ============================================================
# 4. TPS Mapper
# ============================================================
class TPSMapper:
    """
    Thin Plate Spline for 2D point mapping:
        (x1, y1) -> (x2, y2)

    Fit on normalized coordinates.
    """

    def __init__(self, reg: float = 1e-6):
        self.reg = reg
        self.src = None          # (N, 2)
        self.params_x = None     # (N+3,)
        self.params_y = None     # (N+3,)
        self.fitted = False

    @staticmethod
    def _U(r2: np.ndarray) -> np.ndarray:
        """
        Thin plate spline radial basis:
            U(r) = r^2 log(r^2)
        with U(0)=0
        Here input is r^2 for numerical convenience.
        """
        out = np.zeros_like(r2, dtype=np.float64)
        mask = r2 > 1e-20
        out[mask] = r2[mask] * np.log(r2[mask])
        return out

    def fit(self, src_points: np.ndarray, dst_points: np.ndarray):
        """
        src_points: (N, 2) normalized source coords
        dst_points: (N, 2) normalized target coords
        """
        src = np.asarray(src_points, dtype=np.float64)
        dst = np.asarray(dst_points, dtype=np.float64)

        if src.ndim != 2 or src.shape[1] != 2:
            raise ValueError("src_points must have shape (N, 2)")
        if dst.ndim != 2 or dst.shape[1] != 2:
            raise ValueError("dst_points must have shape (N, 2)")
        if len(src) < 3:
            raise ValueError("TPS needs at least 3 points.")
        if len(src) != len(dst):
            raise ValueError("src_points and dst_points must have same length.")

        n = len(src)
        self.src = src

        diff = src[:, None, :] - src[None, :, :]       # (N, N, 2)
        r2 = np.sum(diff ** 2, axis=2)                 # (N, N)
        K = self._U(r2)

        if self.reg > 0:
            K += self.reg * np.eye(n)

        P = np.concatenate([np.ones((n, 1)), src], axis=1)   # (N, 3)

        L_top = np.concatenate([K, P], axis=1)               # (N, N+3)
        L_bottom = np.concatenate([P.T, np.zeros((3, 3))], axis=1)
        L = np.concatenate([L_top, L_bottom], axis=0)        # (N+3, N+3)

        Yx = np.concatenate([dst[:, 0], np.zeros(3)], axis=0)
        Yy = np.concatenate([dst[:, 1], np.zeros(3)], axis=0)

        self.params_x = np.linalg.solve(L, Yx)
        self.params_y = np.linalg.solve(L, Yy)

        self.fitted = True
        return self

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        points: (M, 2) normalized source coords
        returns: (M, 2) normalized target coords
        """
        if not self.fitted:
            raise RuntimeError("TPSMapper must be fitted before transform().")

        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must have shape (M, 2)")

        src = self.src
        n = len(src)
        m = len(pts)

        diff = pts[:, None, :] - src[None, :, :]       # (M, N, 2)
        r2 = np.sum(diff ** 2, axis=2)                 # (M, N)
        K = self._U(r2)                                # (M, N)

        P = np.concatenate([np.ones((m, 1)), pts], axis=1)   # (M, 3)
        A = np.concatenate([K, P], axis=1)                   # (M, N+3)

        x_out = A @ self.params_x
        y_out = A @ self.params_y

        return np.stack([x_out, y_out], axis=1).astype(np.float32)


# ============================================================
# 7. TPS evaluation and inference
# ============================================================
def evaluate_tps(
        tps: TPSMapper,
        src_points_test_px: np.ndarray,
        gt_points_test_px: np.ndarray,
        img1_size: Tuple[int, int],
        img2_size: Tuple[int, int],
):
    src_norm = point_utility.normalize_points(src_points_test_px, img1_size)
    gt_norm = point_utility.normalize_points(gt_points_test_px, img2_size)

    pred_norm = tps.transform(src_norm)
    pred_px = point_utility.denormalize_points(pred_norm, img2_size)
    gt_px = point_utility.denormalize_points(gt_norm, img2_size)

    rmse_px = point_utility.compute_rmse_pixels_from_arrays(pred_px, gt_px)
    mse_norm = float(np.mean((pred_norm - gt_norm) ** 2))
    return mse_norm, rmse_px, pred_norm, gt_norm


def transform_points_tps(
        tps: TPSMapper,
        points_img1: np.ndarray,
        img1_size: Tuple[int, int],
        img2_size: Tuple[int, int],
) -> np.ndarray:
    pts_norm = point_utility.normalize_points(points_img1, img1_size)
    pred_norm = tps.transform(pts_norm)
    pred_px = point_utility.denormalize_points(pred_norm, img2_size)
    return pred_px
