import numpy as np
import objtracker
from objdetector import Detector
import cv2
from ultralytics.hara.hara_deep_sort_HBB.deep_sort.configs.common_cfg import cfg
from deep_sort.deep_sort import DeepSort
from datetime import datetime
import os
from pathlib import Path

def get_deepsort():
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True, use_reid=cfg.DEEPSORT.USE_REID,
                        tracking_csv_path=cfg.DEEPSORT.TRACKING_CSV_PATH)
    return deepsort

def start(show_window=True):
    # Initialize video capture to get video properties
    capture = cv2.VideoCapture(cfg.DEEPSORT.VIDEO_PATH)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {cfg.DEEPSORT.VIDEO_PATH}")

    # Get video properties (width and height)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Close the video capture
    capture.release()

    detector = Detector()
    capture = cv2.VideoCapture(cfg.DEEPSORT.VIDEO_PATH)
    videoWriter = None
    input_video_fps = int(capture.get(cv2.CAP_PROP_FPS))
    print('fps:', input_video_fps)

    # Dictionary to store the trail points of each object
    object_trails = {}
    deepsort = get_deepsort()

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

            _, im = capture.read()
            if im is None:
                break
            # detections = OBBDetections()

            output_image_frame, _ = objtracker.update(detector, im, deepsort)
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

    except Exception as e:
        print(f"An error occurred: {e}")

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
    # cfg.merge_from_file(r"deep_sort/configs/deep_sort.yaml")
    # start()


    # # HBB10
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min\hold1_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min\hold2_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min\hold3_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min\hold4_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min\hold5_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min\hold6_tracking.yaml")
    start(show_window=False)
    #
    #
    # # HBB10_detection
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection\hold1_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection\hold2_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection\hold3_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection\hold4_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection\hold5_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection\hold6_tracking.yaml")
    start(show_window=False)

    #
    # # HBB10_track
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_track\hold1_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_track\hold2_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_track\hold3_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_track\hold4_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_track\hold5_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_track\hold6_tracking.yaml")
    start(show_window=False)
    #
    #
    # # HBB10_detection_track
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection_track\hold1_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection_track\hold2_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection_track\hold3_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection_track\hold4_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection_track\hold5_tracking.yaml")
    start(show_window=False)
    cfg.merge_from_file(r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection_track\hold6_tracking.yaml")
    start(show_window=False)





