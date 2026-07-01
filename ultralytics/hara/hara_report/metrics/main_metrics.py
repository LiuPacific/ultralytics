from ultralytics.hara.hara_report.metrics import detection_identity_counts
from ultralytics.hara.hara_report.metrics import metric_tools




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
    Evaluate tracking metrics and collect detailed count data.

    Returned metrics:
        MOTA, IDS, IDF1

    Returned detection counts:
        TP, FN, FP, TN

    Returned identity counts:
        IDTP, IDFN, IDFP, IDTN
    """
    pred_df, gt_df = detection_identity_counts.load_tracking_data(tracking_csv, gt_csv)

    detection_counts = detection_identity_counts.compute_detection_counts(
        gt_df,
        pred_df,
        distance_threshold=distance_threshold,
        ignore_invisible=ignore_invisible,
        include_undetected_predictions=include_undetected_predictions,
        min_confidence=min_confidence,
        frame_start=frame_start,
        frame_end=frame_end
    )

    identity_counts = detection_identity_counts.compute_identity_counts(
        gt_df,
        pred_df,
        distance_threshold=distance_threshold,
        ignore_invisible=ignore_invisible,
        include_undetected_predictions=include_undetected_predictions,
        min_confidence=min_confidence,
        frame_start=frame_start,
        frame_end=frame_end
    )

    mota = metric_tools.compute_mota(
        gt_df,
        pred_df,
        distance_threshold=distance_threshold,
        ignore_invisible=ignore_invisible,
        include_undetected_predictions=include_undetected_predictions,
        min_confidence=min_confidence,
        frame_start=frame_start,
        frame_end=frame_end
    )

    ids = metric_tools.compute_ids(
        gt_df,
        pred_df,
        distance_threshold=distance_threshold,
        ignore_invisible=ignore_invisible,
        include_undetected_predictions=include_undetected_predictions,
        min_confidence=min_confidence,
        frame_start=frame_start,
        frame_end=frame_end
    )

    idf1 = metric_tools.compute_idf1(
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

        "TP": detection_counts["TP"],
        "FN": detection_counts["FN"],
        "FP": detection_counts["FP"],
        "TN": detection_counts["TN"],

        "IDTP": identity_counts["IDTP"],
        "IDFN": identity_counts["IDFN"],
        "IDFP": identity_counts["IDFP"],
        "IDTN": identity_counts["IDTN"]
    }

# ============================================================
# 7. Example usage
# ============================================================

if __name__ == "__main__":
    # For original model,

    # gt_csv = r".run/08_mock_1-15_tracking.csv"
    # tracking_csv = r".run3/obb10hold1_tracking.csv"
    # tracking_csv = r".run3/obb10hold1_reid_tracking.csv"
    # tracking_csv = r".run4/opt_hold1_10min_noinit_tracking_output_07.csv"
    # tracking_csv = r".run4/opt10hold1_noinit_reid_tracking.csv"
    # tracking_csv = r".run3/opt_hold1_10min_tracking_output_07.csv"
    # tracking_csv = r".run3/opt10hold1_reid_tracking.csv"
    #


    # gt_csv = r".run\08_sick_16-30_tracking.csv"
    # tracking_csv = r".run3/obb10hold2_tracking.csv"
    # tracking_csv = r".run3/obb10hold2_reid_tracking.csv"
    # tracking_csv = r".run4/opt_hold2_10min_noinit_tracking_output_07.csv"
    # tracking_csv = r".run4/opt10hold2_noinit_reid_tracking.csv"
    # tracking_csv = r".run3/opt_hold2_10min_tracking_output_07.csv"
    # tracking_csv = r".run3/opt10hold2_reid_tracking.csv"

    # gt_csv = r".run\10_mock_76-90_tracking.csv"
    # tracking_csv = r".run3/obb10hold3_tracking.csv"
    # tracking_csv = r".run2/obb10hold3_reid_tracking.csv"
    # tracking_csv = r".run4/opt_hold3_10min_noinit_tracking_output_07.csv"
    # tracking_csv = r".run4/opt10hold3_noinit_reid_tracking.csv"
    # tracking_csv = r".run3/opt_hold3_10min_tracking_output_07.csv"
    # tracking_csv = r".run2/opt10hold3_reid_tracking.csv"

    # gt_csv = r".run\10_sick_31-45_tracking.csv"
    # tracking_csv = r".run3/obb10hold4_tracking.csv"
    # tracking_csv = r".run2/obb10hold4_reid_tracking.csv"
    # tracking_csv = r".run4/opt_hold4_10min_noinit_tracking_output_07.csv"
    # tracking_csv = r".run4/opt10hold4_noinit_reid_tracking.csv"
    # tracking_csv = r".run3/opt_hold4_10min_tracking_output_07.csv"
    # tracking_csv = r".run2/opt10hold4_reid_tracking.csv"

    # gt_csv = r".run\12_mock_46-60_tracking.csv"
    # tracking_csv = r".run3/obb10hold5_tracking.csv"
    # tracking_csv = r".run2/obb10hold5_reid_tracking.csv"
    # tracking_csv = r".run4/opt_hold5_10min_noinit_tracking_output_07.csv"
    # tracking_csv = r".run4/opt10hold5_noinit_reid_tracking.csv"
    # tracking_csv = r".run3/opt_hold5_10min_tracking_output_07.csv"
    # tracking_csv = r".run2/opt10hold5_reid_tracking.csv"

    # gt_csv = r".run\12_sick_61-75_tracking.csv"
    # tracking_csv = r".run3/obb10hold6_tracking.csv"
    # tracking_csv = r".run2/obb10hold6_reid_tracking.csv"
    # tracking_csv = r".run4/opt_hold6_10min_noinit_tracking_output_07.csv"
    # tracking_csv = r".run4/opt10hold6_noinit_reid_tracking.csv"
    # tracking_csv = r".run3/opt_hold6_10min_tracking_output_07.csv"
    # tracking_csv = r".run2/opt10hold6_reid_tracking.csv"


    metrics = evaluate_tracking(
        tracking_csv=tracking_csv,
        gt_csv=gt_csv,
        distance_threshold=50,
        ignore_invisible=False,
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












