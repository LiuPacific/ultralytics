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

def run_one():

    gt_csv = r".run/08_mock_1-15_tracking.csv"
    tracking_csv = r".run10/hbb_l_10hold1.csv"
    # tracking_csv = r".run6/obb10hold1_reid_10.csv"
    # tracking_csv = r".run6/obb10hold1_detection_track_10.csv"
    # tracking_csv = r".run6/obb10hold1_detection_track_reid_10.csv"
    # tracking_csv = r".run6/obb10hold1_detection_track_reuseID_10.csv"
    # tracking_csv = r".run6/obb10hold1_detection_track_reuseID_reid_10.csv"
    # tracking_csv = r".run6/obb10hold1_detection_10.csv"
    # tracking_csv = r".run6/obb10hold1_reuseID_10.csv"
    # tracking_csv = r".run6/obb10hold1_track_10.csv"

    # gt_csv = r".run/08_sick_16-30_tracking.csv"
    # tracking_csv = r".run9_hbb/hbb10hold2.csv"
    # tracking_csv = r".run6/obb10hold2_reid_10.csv"
    # tracking_csv = r".run6/obb10hold2_detection_track_10.csv"
    # tracking_csv = r".run6/obb10hold2_detection_track_reid_10.csv"
    # tracking_csv = r".run6/obb10hold2_detection_track_reuseID_10.csv"
    # tracking_csv = r".run6/obb10hold2_detection_track_reuseID_reid_10.csv"
    # tracking_csv = r".run6/obb10hold2_detection_10.csv"
    # tracking_csv = r".run6/obb10hold2_reuseID_10.csv"
    # tracking_csv = r".run6/obb10hold2_track_10.csv"

    # gt_csv = r".run\10_mock_76-90_tracking_10.csv"
    # tracking_csv = r".run6/obb10hold3_10.csv"
    # tracking_csv = r".run6/obb10hold3_reid_10.csv"
    # tracking_csv = r".run6/obb10hold3_detection_track_10.csv"
    # tracking_csv = r".run6/obb10hold3_detection_track_reid_10.csv"
    # tracking_csv = r".run6/obb10hold3_detection_track_reuseID_10.csv"
    # tracking_csv = r".run6/obb10hold3_detection_track_reuseID_reid_10.csv"
    # tracking_csv = r".run6/obb10hold3_detection_10.csv"
    # tracking_csv = r".run6/obb10hold3_reuseID_10.csv"
    # tracking_csv = r".run6/obb10hold3_track_10.csv"

    # gt_csv = r".run\10_sick_31-45_tracking.csv"
    # tracking_csv = r".run9_hbb/hbb10hold4.csv"
    # tracking_csv = r".run9_hbb/hbb10hold4_detection.csv"
    # tracking_csv = r".run9_hbb/hbb10hold4_detection_track.csv"
    # tracking_csv = r".run9_hbb/hbb10hold4_track.csv"

    # gt_csv = r".run\12_mock_46-60_tracking_10.csv"
    # tracking_csv = r".run6/obb10hold5_10.csv"
    # tracking_csv = r".run6/obb10hold5_reid_10.csv"
    # tracking_csv = r".run6/obb10hold5_detection_track_10.csv"
    # tracking_csv = r".run6/obb10hold5_detection_track_reid_10.csv"
    # tracking_csv = r".run6/obb10hold5_detection_track_reuseID_10.csv"
    # tracking_csv = r".run6/obb10hold5_detection_track_reuseID_reid_10.csv"
    # tracking_csv = r".run6/obb10hold5_detection_10.csv"
    # tracking_csv = r".run6/obb10hold5_reuseID_10.csv"
    # tracking_csv = r".run6/obb10hold5_track_10.csv"

    # gt_csv = r".run\12_sick_61-75_tracking_10.csv"
    # tracking_csv = r".run6/obb10hold6_10.csv"
    # tracking_csv = r".run6/obb10hold6_reid_10.csv"
    # tracking_csv = r".run6/obb10hold6_detection_track_10.csv"
    # tracking_csv = r".run6/obb10hold6_detection_track_reid_10.csv"
    # tracking_csv = r".run6/obb10hold6_detection_track_reuseID_10.csv"
    # tracking_csv = r".run6/obb10hold6_detection_track_reuseID_reid_10.csv"
    # tracking_csv = r".run6/obb10hold6_detection_10.csv"
    # tracking_csv = r".run6/obb10hold6_reuseID_10.csv"
    # tracking_csv = r".run6/obb10hold6_track_10.csv"
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

def run_batch():

    # gt_csv = r".run/08_mock_1-15_tracking.csv"
    # tracking_csv_s = [
    #     r".run10/hbb_l_10hold1.csv",
    #     r".run10/hbb_l_10hold1_detection.csv",
    #     r".run10/hbb_l_10hold1_track.csv",
    #     r".run10/hbb_l_10hold1_detection_track.csv",
    #     r".run10/obb_l_10hold1.csv",
    #     r".run10/obb_l_10hold1_reid.csv",
    #     r".run10/obb_l_10hold1_detection.csv",
    #     r".run10/obb_l_10hold1_track.csv",
    #     r".run10/obb_l_10hold1_detection_track.csv",
    #     r".run10/obb_l_10hold1_detection_track_reid.csv",
    #     r".run10/hbb_l_10hold1_bytetrack.csv",
    #     r".run10/hbb_l_10hold1_botsort.csv",
    # ]
    #
    # gt_csv = r".run\08_sick_16-30_tracking.csv"
    # tracking_csv_s = [
    #     r".run10/hbb_l_10hold2.csv",
    #     r".run10/hbb_l_10hold2_detection.csv",
    #     r".run10/hbb_l_10hold2_track.csv",
    #     r".run10/hbb_l_10hold2_detection_track.csv",
    #     r".run10/obb_l_10hold2.csv",
    #     r".run10/obb_l_10hold2_reid.csv",
    #     r".run10/obb_l_10hold2_detection.csv",
    #     r".run10/obb_l_10hold2_track.csv",
    #     r".run10/obb_l_10hold2_detection_track.csv",
    #     r".run10/obb_l_10hold2_detection_track_reid.csv",
    #     r".run10/hbb_l_10hold2_bytetrack.csv",
    #     r".run10/hbb_l_10hold2_botsort.csv",
    # ]
    #
    # gt_csv = r".run\10_mock_76-90_tracking.csv"
    # tracking_csv_s = [
    #     r".run10/hbb_l_10hold3.csv",
    #     r".run10/hbb_l_10hold3_detection.csv",
    #     r".run10/hbb_l_10hold3_track.csv",
    #     r".run10/hbb_l_10hold3_detection_track.csv",
    #     r".run10/obb_l_10hold3.csv",
    #     r".run10/obb_l_10hold3_reid.csv",
    #     r".run10/obb_l_10hold3_detection.csv",
    #     r".run10/obb_l_10hold3_track.csv",
    #     r".run10/obb_l_10hold3_detection_track.csv",
    #     r".run10/obb_l_10hold3_detection_track_reid.csv",
    #     r".run10/hbb_l_10hold3_bytetrack.csv",
    #     r".run10/hbb_l_10hold3_botsort.csv",
    # ]
    #
    # gt_csv = r".run\10_sick_31-45_tracking.csv"
    # tracking_csv_s = [
    #     r".run10/hbb_l_10hold4.csv",
    #     r".run10/hbb_l_10hold4_detection.csv",
    #     r".run10/hbb_l_10hold4_track.csv",
    #     r".run10/hbb_l_10hold4_detection_track.csv",
    #     r".run10/obb_l_10hold4.csv",
    #     r".run10/obb_l_10hold4_reid.csv",
    #     r".run10/obb_l_10hold4_detection.csv",
    #     r".run10/obb_l_10hold4_track.csv",
    #     r".run10/obb_l_10hold4_detection_track.csv",
    #     r".run10/obb_l_10hold4_detection_track_reid.csv",
    #     r".run10/hbb_l_10hold4_bytetrack.csv",
    #     r".run10/hbb_l_10hold4_botsort.csv",
    # ]
    #
    # gt_csv = r".run\12_mock_46-60_tracking.csv"
    # tracking_csv_s = [
    #     r".run10/hbb_l_10hold5.csv",
    #     r".run10/hbb_l_10hold5_detection.csv",
    #     r".run10/hbb_l_10hold5_track.csv",
    #     r".run10/hbb_l_10hold5_detection_track.csv",
    #     r".run10/obb_l_10hold5.csv",
    #     r".run10/obb_l_10hold5_reid.csv",
    #     r".run10/obb_l_10hold5_detection.csv",
    #     r".run10/obb_l_10hold5_track.csv",
    #     r".run10/obb_l_10hold5_detection_track.csv",
    #     r".run10/obb_l_10hold5_detection_track_reid.csv",
    #     r".run10/hbb_l_10hold5_bytetrack.csv",
    #     r".run10/hbb_l_10hold5_botsort.csv",
    # ]
    #
    gt_csv = r".run\12_sick_61-75_tracking.csv"
    tracking_csv_s = [
        r".run10/hbb_l_10hold6.csv",
        r".run10/hbb_l_10hold6_detection.csv",
        r".run10/hbb_l_10hold6_track.csv",
        r".run10/hbb_l_10hold6_detection_track.csv",
        r".run10/obb_l_10hold6.csv",
        r".run10/obb_l_10hold6_reid.csv",
        r".run10/obb_l_10hold6_detection.csv",
        r".run10/obb_l_10hold6_track.csv",
        r".run10/obb_l_10hold6_detection_track.csv",
        r".run10/obb_l_10hold6_detection_track_reid.csv",
        r".run10/hbb_l_10hold6_bytetrack.csv",
        r".run10/hbb_l_10hold6_botsort.csv",
    ]

    for tracking_csv in tracking_csv_s:
        print(f"Evaluating tracking CSV: {tracking_csv}")
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

        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("---------------------------")

if __name__ == "__main__":
    # run_one()
    run_batch()












