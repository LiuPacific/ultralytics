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

def make_side_by_side(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Resize image2 height to image1 height for side-by-side visualization.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    scale = h1 / h2
    new_w2 = int(w2 * scale)
    img2_resized = cv2.resize(img2, (new_w2, h1))

    canvas = np.zeros((h1, w1 + new_w2, 3), dtype=np.uint8)
    canvas[:, :w1] = img1
    canvas[:, w1:w1 + new_w2] = img2_resized

    return canvas, scale

def draw_point_pairs(
        img1: np.ndarray,
        img2: np.ndarray,
        src_points: np.ndarray,
        pred_points_img2: np.ndarray,
        gt_points_img2: Optional[np.ndarray] = None,
        output_path: str = "comparison.png",
        label_prefix: str = "pt"
):
    """
    Draw source points on image1 and predicted / GT points on image2.
    A side-by-side comparison figure is saved.

    Blue circles  : source points in image1
    Green circles : predicted mapped points in image2
    Red circles   : ground-truth points in image2 (if provided)
    Yellow line   : prediction-to-ground-truth error line (if provided)
    White line    : link from left image point to right image predicted point
    """
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    canvas, scale = make_side_by_side(img1, img2)
    h1, w1 = img1.shape[:2]

    pred_points_scaled = pred_points_img2.copy().astype(np.float32)
    pred_points_scaled[:, 0] *= scale
    pred_points_scaled[:, 1] *= scale

    gt_points_scaled = None
    if gt_points_img2 is not None:
        gt_points_scaled = gt_points_img2.copy().astype(np.float32)
        gt_points_scaled[:, 0] *= scale
        gt_points_scaled[:, 1] *= scale

    for i, src in enumerate(src_points):
        x1, y1 = int(round(src[0])), int(round(src[1]))
        px2, py2 = int(round(pred_points_scaled[i, 0] + w1)), int(round(pred_points_scaled[i, 1]))

        # source point
        cv2.circle(canvas, (x1, y1), 6, (255, 0, 0), -1)  # blue

        # predicted point
        cv2.circle(canvas, (px2, py2), 6, (0, 255, 0), -1)  # green

        # line from source to predicted target
        cv2.line(canvas, (x1, y1), (px2, py2), (255, 255, 255), 1)

        # optional GT
        if gt_points_scaled is not None:
            gx2, gy2 = int(round(gt_points_scaled[i, 0] + w1)), int(round(gt_points_scaled[i, 1]))
            cv2.circle(canvas, (gx2, gy2), 6, (0, 0, 255), 2)  # red
            cv2.line(canvas, (px2, py2), (gx2, gy2), (0, 255, 255), 1)  # yellow error line

        cv2.putText(
            canvas,
            f"{label_prefix}{i}",
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_path, canvas)
    print(f"Saved visualization to: {output_path}")


if __name__ == '__main__':
    csv_path = r"/ultralytics/hara/tools/pose/video_tools/registrar/.data/readme.csv"
    image1_path = r"/ultralytics/hara/tools/pose/video_tools/registrar/.data/rgb_frame_00_00_01.png"
    image2_path = r"/ultralytics/hara/tools/pose/video_tools/registrar/.data/thermal_00_00_01.png"

    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    df = pd.read_csv(csv_path)

    left_points = df[["x1", "y1"]].values.astype(np.float32)
    right_points = df[["x2", "y2"]].values.astype(np.float32)


    draw_point_pairs(
        img1=img1,
        img2=img2,
        src_points=left_points,
        pred_points_img2=right_points,
        gt_points_img2=right_points, # here I put the prediction adn ground truth the same.
        output_path="../test.png",
        label_prefix="t"
    )
