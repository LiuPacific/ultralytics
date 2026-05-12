import os
import cv2
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


def compute_rmse_pixels_from_arrays(pred_px: np.ndarray, gt_px: np.ndarray) -> float:
    pred_px = np.asarray(pred_px, dtype=np.float64)
    gt_px = np.asarray(gt_px, dtype=np.float64)
    return float(np.sqrt(np.mean((pred_px - gt_px) ** 2)))


def normalize_points(points_px: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    w, h = img_size
    pts = points_px.astype(np.float32).copy()
    pts[:, 0] /= w
    pts[:, 1] /= h
    return pts


def denormalize_points(points_norm: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    w, h = img_size
    pts = points_norm.astype(np.float32).copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts
