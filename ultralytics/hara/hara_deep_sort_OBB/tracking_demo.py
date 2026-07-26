import cv2
import os
import yaml
from datetime import datetime
from obj_obb_detector import ObbDetector
import objtracker
from ultralytics.hara.hara_deep_sort_OBB.deep_sort.configs.common_cfg import cfg
from ultralytics.hara.hara_deep_sort_OBB.deep_sort.deep_sort.deep_sort_obb import DeepSORTOBB
from pathlib import Path



def get_deepsort_obb():
    deepsort_obb = DeepSORTOBB(cfg.DEEPSORT.REID_CKPT,
                               max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                               max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                               use_cuda=True,
                               MAX_ID_POOL=cfg.DEEPSORT.MAX_ID_POOL,
                               reconnection_distance_threshold=cfg.DEEPSORT.RECONNECTION_DISTANCE_THRESHOLD,
                               reuse_id_assignment_distance_threshold=cfg.DEEPSORT.REUSE_ID_ASSIGNMENT_DISTANCE_THRESHOLD,
                               use_reid=cfg.DEEPSORT.USE_REID,
                               use_rotated_features=cfg.DEEPSORT.USE_ROTATED_FEATURES,
                               tracking_csv_path=cfg.DEEPSORT.TRACKING_CSV_PATH
                               )
    return deepsort_obb

def start(show_window=True):
    # Initialize video capture to get video properties
    capture = cv2.VideoCapture(cfg.DEEPSORT.VIDEO_PATH)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {cfg.DEEPSORT.VIDEO_PATH}")

    # Get video properties (width and height)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


    # Close the video capture (we'll re-open for actual processing)
    capture.release()

    detector = ObbDetector()
    capture = cv2.VideoCapture(cfg.DEEPSORT.VIDEO_PATH)
    videoWriter = None
    input_video_fps = int(capture.get(cv2.CAP_PROP_FPS))
    print('input video fps:', input_video_fps)

    deepsort_obb = get_deepsort_obb()

    global_frame_id = 0
    max_frame = cfg.DEEPSORT.MAX_FRAME

    # Logging / summary file
    start_time = datetime.now()
    frames_processed = 0
    out_filename = f"output_{start_time.strftime('%Y%m%d_%H%M%S')}_{Path(cfg.DEEPSORT.get('RESULT_PATH', "temp.mp4")).stem}.txt"
    out_filepath = os.path.join(os.getcwd(), out_filename)

    try:
        while True:
            if global_frame_id > max_frame:
                break
            global_frame_id += 1

            ret, im = capture.read()
            if not ret or im is None:
                break

            # Process frame
            output_image_frame, _ = objtracker.update(detector, im, deepsort_obb)
            frames_processed += 1

            if videoWriter is None:
                fourcc = cv2.VideoWriter_fourcc(
                    'm', 'p', '4', 'v')  # opencv3.0
                videoWriter = cv2.VideoWriter(
                    cfg.DEEPSORT.RESULT_PATH, fourcc, input_video_fps, (output_image_frame.shape[1], output_image_frame.shape[0]))

            videoWriter.write(output_image_frame)

            if show_window:
                height, width = output_image_frame.shape[:2]
                output_image_frame = cv2.resize(output_image_frame, (int(width/3), int(height/3)))
                cv2.imshow('Demo', output_image_frame)
                cv2.waitKey(1)
    finally:
        # cleanup
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        processing_fps = frames_processed / elapsed if elapsed > 0 else 0.0

        # Write summary to file
        try:
            with open(out_filepath, 'w', encoding='utf-8') as f:
                f.write(f"start_time: {start_time.isoformat()}\n")
                f.write(f"end_time: {end_time.isoformat()}\n")
                f.write(f"elapsed_seconds: {elapsed:.3f}\n")
                f.write(f"input_video_fps: {input_video_fps}\n")
                f.write(f"processing_fps: {processing_fps:.3f}\n")
                f.write(f"frames_processed: {frames_processed}\n")
                f.write(f"RESULT_PATH: {cfg.DEEPSORT.get("RESULT_PATH")}\n")
                f.write(f"deepsort_cfg: {cfg.get("DEEPSORT")}\n")
            print(f"Wrote tracking summary to {out_filepath}")
        except Exception as e:
            print(f"Failed to write summary file {out_filepath}: {e}")

        try:
            capture.release()
        except Exception:
            pass
        try:
            if videoWriter is not None:
                videoWriter.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
    cfg.merge_from_file("deep_sort/configs/hold1_tracking_det.yaml")
    start(show_window=True)



    # #
    # # OBB10
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min\hold1_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min\hold2_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min\hold3_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min\hold4_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min\hold5_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min\hold6_tracking.yaml")
    # start(show_window=False)
    #
    # #
    # # OBB10 detection
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection\hold1_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection\hold2_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection\hold3_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection\hold4_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection\hold5_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection\hold6_tracking.yaml")
    # start(show_window=False)
    #
    # #
    # # OBB10 track
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_track\hold1_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_track\hold2_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_track\hold3_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_track\hold4_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_track\hold5_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_track\hold6_tracking.yaml")
    # start(show_window=False)
    #
    # #
    # # OBB10 detection track
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection_track\hold1_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection_track\hold2_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection_track\hold3_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection_track\hold4_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection_track\hold5_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection_track\hold6_tracking.yaml")
    # start(show_window=False)
    # #
    # # #
    # # # OBB10minReID
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_ReID\hold1_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_ReID\hold2_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_ReID\hold3_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_ReID\hold4_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_ReID\hold5_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_ReID\hold6_tracking.yaml")
    # start(show_window=False)
    #
    # #
    # # Opt10min_detection_track reID
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection_track_ReID\hold1_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection_track_ReID\hold2_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection_track_ReID\hold3_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection_track_ReID\hold4_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection_track_ReID\hold5_tracking.yaml")
    # start(show_window=False)
    # cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\OBB10min_detection_track_ReID\hold6_tracking.yaml")
    # start(show_window=False)
