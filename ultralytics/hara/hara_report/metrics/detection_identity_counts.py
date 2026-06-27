from ultralytics.hara.hara_report.metrics import detection_identity_counts
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ============================================================
# 1. Data preparation
# ============================================================

def load_tracking_data(tracking_csv, gt_csv):
    """
    Load prediction tracking output and ground truth CSV files.
    """
    pred_df = pd.read_csv(tracking_csv)
    gt_df = pd.read_csv(gt_csv)

    return pred_df, gt_df


def prepare_predictions(
        pred_df,
        include_undetected_predictions=True,
        min_confidence=None
):
    """
    Prepare tracker output.

    Parameters
    ----------
    include_undetected_predictions:
        True  -> use all tracking rows.
        False -> only use rows where detected == True.

    min_confidence:
        Optional confidence threshold.
    """
    pred_df = pred_df.copy()

    if not include_undetected_predictions and "detected" in pred_df.columns:
        pred_df = pred_df[pred_df["detected"] == True]

    if min_confidence is not None and "confidence" in pred_df.columns:
        pred_df = pred_df[pred_df["confidence"] >= min_confidence]

    return pred_df


def filter_by_frame_range(
        gt_df,
        pred_df,
        frame_start=None,
        frame_end=None
):
    """
    Filter GT and prediction data by global frame id range.

    GT uses column: frame
    Prediction uses column: frame_id

    The range is inclusive:
        frame_start <= frame <= frame_end
    """
    gt_df = gt_df.copy()
    pred_df = pred_df.copy()

    if frame_start is not None:
        gt_df = gt_df[gt_df["frame"] >= frame_start]
        pred_df = pred_df[pred_df["frame_id"] >= frame_start]

    if frame_end is not None:
        gt_df = gt_df[gt_df["frame"] <= frame_end]
        pred_df = pred_df[pred_df["frame_id"] <= frame_end]

    return gt_df, pred_df

def build_matches(
        gt_df,
        pred_df,
        distance_threshold=100,
        ignore_invisible=False,
        include_undetected_predictions=True,
        min_confidence=None,
        frame_start=None,
        frame_end=None
):
    """
    Build all frame-level GT-track matches.

    Only frames inside [frame_start, frame_end] are evaluated.

    Important:
    Since your GT is only labeled on some frames, this function only evaluates
    frames that exist in the GT file.
    """
    gt_df = prepare_ground_truth(gt_df, ignore_invisible=ignore_invisible)

    pred_df = prepare_predictions(
        pred_df,
        include_undetected_predictions=include_undetected_predictions,
        min_confidence=min_confidence
    )

    gt_df, pred_df = filter_by_frame_range(
        gt_df,
        pred_df,
        frame_start=frame_start,
        frame_end=frame_end
    )

    gt_frames = sorted(gt_df["frame"].unique())

    all_matches = []

    total_gt = 0
    total_pred = 0
    total_fn = 0
    total_fp = 0

    for frame in gt_frames:
        gt_frame_df = gt_df[gt_df["frame"] == frame]
        pred_frame_df = pred_df[pred_df["frame_id"] == frame]

        total_gt += len(gt_frame_df)
        total_pred += len(pred_frame_df)

        frame_matches = match_one_frame(
            gt_frame_df,
            pred_frame_df,
            distance_threshold
        )

        num_matches = len(frame_matches)

        total_fn += len(gt_frame_df) - num_matches
        total_fp += len(pred_frame_df) - num_matches

        for m in frame_matches:
            m["frame"] = frame
            all_matches.append(m)

    matches_df = pd.DataFrame(all_matches)

    return {
        "matches": matches_df,
        "total_gt": total_gt,
        "total_pred": total_pred,
        "fn": total_fn,
        "fp": total_fp
    }


def prepare_ground_truth(
        gt_df,
        ignore_invisible=False
):
    """
    Prepare ground truth.

    Parameters
    ----------
    ignore_invisible:
        False -> invisible chickens are still counted as GT objects.
                 This is useful if you want to evaluate occlusion tracking.
        True  -> invisible chickens are ignored.
    """
    gt_df = gt_df.copy()

    if ignore_invisible and "invisible" in gt_df.columns:
        gt_df = gt_df[gt_df["invisible"] == 0]

    return gt_df


# ============================================================
# 2. Frame-level matching
# ============================================================

def match_one_frame(
        gt_frame_df,
        pred_frame_df,
        distance_threshold
):
    """
    Match GT chickens and predicted tracks in one frame using Hungarian matching.

    Matching is based on Euclidean distance between:
        GT center:   x, y
        Pred center: center_x, center_y

    Returns
    -------
    matches : list of dict
        Each dict contains chicken_id, track_id, and distance.
    """
    matches = []

    if len(gt_frame_df) == 0 or len(pred_frame_df) == 0:
        return matches

    gt_points = gt_frame_df[["x", "y"]].to_numpy(dtype=float)
    pred_points = pred_frame_df[["center_x", "center_y"]].to_numpy(dtype=float)

    # Distance matrix: shape = [num_gt, num_pred]
    dist_matrix = np.linalg.norm(
        gt_points[:, None, :] - pred_points[None, :, :],
        axis=2
    )

    # Large cost for impossible matches
    large_cost = 1e9
    cost_matrix = dist_matrix.copy()
    cost_matrix[cost_matrix > distance_threshold] = large_cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    for r, c in zip(row_ind, col_ind):
        distance = dist_matrix[r, c]

        if distance <= distance_threshold:
            matches.append({
                "chicken_id": gt_frame_df.iloc[r]["chicken_id"],
                "track_id": pred_frame_df.iloc[c]["track_id"],
                "distance": distance
            })

    return matches

def compute_detection_counts(
        gt_df,
        pred_df,
        distance_threshold=100,
        ignore_invisible=False,
        include_undetected_predictions=True,
        min_confidence=None,
        frame_start=None,
        frame_end=None
):
    """
    Compute detection-level TP, FN, FP, TN.

    TP:
        A GT chicken is matched to one predicted track within distance_threshold.

    FN:
        A GT chicken exists, but no prediction is matched to it.

    FP:
        A predicted track exists, but it is not matched to any GT chicken.

    TN:
        Not well-defined in object detection / MOT.
        There is no fixed number of true negative background objects.
        Therefore, TN is returned as np.nan.
    """
    result = detection_identity_counts.build_matches(
        gt_df,
        pred_df,
        distance_threshold=distance_threshold,
        ignore_invisible=ignore_invisible,
        include_undetected_predictions=include_undetected_predictions,
        min_confidence=min_confidence,
        frame_start=frame_start,
        frame_end=frame_end
    )

    tp = len(result["matches"])
    fn = result["fn"]
    fp = result["fp"]
    tn = np.nan

    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn
    }


def compute_identity_counts(
        gt_df,
        pred_df,
        distance_threshold=100,
        ignore_invisible=False,
        include_undetected_predictions=True,
        min_confidence=None,
        frame_start=None,
        frame_end=None
):
    """
    Compute identity-level IDTP, IDFN, IDFP, IDTN.

    IDTP:
        Number of matched detections whose assigned track_id is globally
        consistent with one GT chicken_id.

    IDFN:
        GT identity observations not correctly identified.

    IDFP:
        Predicted identity observations not correctly assigned.

    IDTN:
        Not well-defined in IDF1 / MOT identity evaluation.
        Therefore, IDTN is returned as np.nan.
    """
    result = detection_identity_counts.build_matches(
        gt_df,
        pred_df,
        distance_threshold=distance_threshold,
        ignore_invisible=ignore_invisible,
        include_undetected_predictions=include_undetected_predictions,
        min_confidence=min_confidence,
        frame_start=frame_start,
        frame_end=frame_end
    )

    matches_df = result["matches"]
    total_gt = result["total_gt"]
    total_pred = result["total_pred"]

    if matches_df.empty:
        return {
            "IDTP": 0,
            "IDFN": total_gt,
            "IDFP": total_pred,
            "IDTN": np.nan
        }

    identity_count_matrix = pd.crosstab(
        matches_df["chicken_id"],
        matches_df["track_id"]
    )

    count_matrix = identity_count_matrix.to_numpy()

    # Hungarian assignment to maximize identity-consistent matches
    row_ind, col_ind = linear_sum_assignment(-count_matrix)

    idtp = count_matrix[row_ind, col_ind].sum()
    idfn = total_gt - idtp
    idfp = total_pred - idtp
    idtn = np.nan

    return {
        "IDTP": int(idtp),
        "IDFN": int(idfn),
        "IDFP": int(idfp),
        "IDTN": idtn
    }