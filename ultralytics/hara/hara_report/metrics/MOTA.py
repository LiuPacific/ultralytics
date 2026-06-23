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


# ============================================================
# 3. IDS
# ============================================================
def compute_ids(
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
    Compute ID Switches, IDS.

    IDS is counted when the same GT chicken_id is matched to a different
    track_id than before within the selected frame range.
    """
    result = build_matches(
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

    if matches_df.empty:
        return 0

    ids = 0

    matches_df = matches_df.sort_values(["chicken_id", "frame"])

    for chicken_id, group in matches_df.groupby("chicken_id"):
        previous_track_id = None

        for _, row in group.iterrows():
            current_track_id = row["track_id"]

            if previous_track_id is not None:
                if current_track_id != previous_track_id:
                    ids += 1

            previous_track_id = current_track_id

    return ids

# ============================================================
# 4. MOTA
# ============================================================
def compute_mota(
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
    Compute MOTA within a selected global frame id range.

    Formula:
        MOTA = 1 - (FN + FP + IDS) / total_GT
    """
    result = build_matches(
        gt_df,
        pred_df,
        distance_threshold=distance_threshold,
        ignore_invisible=ignore_invisible,
        include_undetected_predictions=include_undetected_predictions,
        min_confidence=min_confidence,
        frame_start=frame_start,
        frame_end=frame_end
    )

    total_gt = result["total_gt"]
    fn = result["fn"]
    fp = result["fp"]

    ids = compute_ids(
        gt_df,
        pred_df,
        distance_threshold=distance_threshold,
        ignore_invisible=ignore_invisible,
        include_undetected_predictions=include_undetected_predictions,
        min_confidence=min_confidence,
        frame_start=frame_start,
        frame_end=frame_end
    )

    if total_gt == 0:
        return np.nan

    mota = 1.0 - (fn + fp + ids) / total_gt

    return mota


# ============================================================
# 5. IDF1
# ============================================================
def compute_idf1(
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
    Compute IDF1 within a selected global frame id range.

    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    """
    result = build_matches(
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
        return 0.0

    identity_count_matrix = pd.crosstab(
        matches_df["chicken_id"],
        matches_df["track_id"]
    )

    count_matrix = identity_count_matrix.to_numpy()

    row_ind, col_ind = linear_sum_assignment(-count_matrix)

    idtp = count_matrix[row_ind, col_ind].sum()

    idfn = total_gt - idtp
    idfp = total_pred - idtp

    denominator = 2 * idtp + idfp + idfn

    if denominator == 0:
        return 0.0

    idf1 = 2 * idtp / denominator

    return idf1


# ============================================================
# 6. Optional: evaluate all metrics together
# ============================================================
def evaluate_tracking(
        tracking_csv,
        gt_csv,
        distance_threshold=100,
        ignore_invisible=False,
        include_undetected_predictions=True,
        min_confidence=None,
        frame_start=None,
        frame_end=None
):
    """
    Evaluate MOTA, IDS, and IDF1 together within a selected frame range.
    """
    pred_df, gt_df = load_tracking_data(tracking_csv, gt_csv)

    result = build_matches(
        gt_df,
        pred_df,
        distance_threshold=distance_threshold,
        ignore_invisible=ignore_invisible,
        include_undetected_predictions=include_undetected_predictions,
        min_confidence=min_confidence,
        frame_start=frame_start,
        frame_end=frame_end
    )

    mota = compute_mota(
        gt_df,
        pred_df,
        distance_threshold=distance_threshold,
        ignore_invisible=ignore_invisible,
        include_undetected_predictions=include_undetected_predictions,
        min_confidence=min_confidence,
        frame_start=frame_start,
        frame_end=frame_end
    )

    ids = compute_ids(
        gt_df,
        pred_df,
        distance_threshold=distance_threshold,
        ignore_invisible=ignore_invisible,
        include_undetected_predictions=include_undetected_predictions,
        min_confidence=min_confidence,
        frame_start=frame_start,
        frame_end=frame_end
    )

    idf1 = compute_idf1(
        gt_df,
        pred_df,
        distance_threshold=distance_threshold,
        ignore_invisible=ignore_invisible,
        include_undetected_predictions=include_undetected_predictions,
        min_confidence=min_confidence,
        frame_start=frame_start,
        frame_end=frame_end
    )

    return {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "MOTA": mota,
        "IDS": ids,
        "IDF1": idf1,
        "FN": result["fn"],
        "FP": result["fp"],
        "total_GT": result["total_gt"],
        "total_predictions_on_GT_frames": result["total_pred"],
        "num_matches": len(result["matches"])
    }

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

# ============================================================
# 7. Example usage
# ============================================================

if __name__ == "__main__":
    tracking_csv = r".run/obb_hold1_10min_tracking_output.csv"
    gt_csv = r".run\08_mock_1-15_tracking.csv"


    metrics = evaluate_tracking(
        tracking_csv=tracking_csv,
        gt_csv=gt_csv,
        distance_threshold=50,
        ignore_invisible=True,
        include_undetected_predictions=False,
        min_confidence=None,

        # Only evaluate this global frame id range
        frame_start=0,
        frame_end=18060 # 10:02
    )

    print("Tracking Evaluation Results")
    print("---------------------------")
    for k, v in metrics.items():
        print(f"{k}: {v}")