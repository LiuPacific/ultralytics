import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from ultralytics.hara.hara_report.metrics import detection_identity_counts



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

    Formula:
        IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    """
    id_counts = compute_identity_counts(
        gt_df,
        pred_df,
        distance_threshold=distance_threshold,
        ignore_invisible=ignore_invisible,
        include_undetected_predictions=include_undetected_predictions,
        min_confidence=min_confidence,
        frame_start=frame_start,
        frame_end=frame_end
    )

    idtp = id_counts["IDTP"]
    idfn = id_counts["IDFN"]
    idfp = id_counts["IDFP"]

    denominator = 2 * idtp + idfp + idfn

    if denominator == 0:
        return 0.0

    return 2 * idtp / denominator