import numpy as np
import cv2
from yolov11tracker import yolov11Tracker
from datetime import datetime
import os
from pathlib import Path
import yaml
from ultralytics.hara.hara_bot_sort.config.common_cfg import cfg


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Detections:
    def __init__(self):
        self.detections = []

    def add(self, xyxy, confidence, class_id, tracker_id):
        self.detections.append((xyxy, confidence, class_id, tracker_id))


# Dark BGR colors for around 15 tracks
DARK_COLORS = [
    (0, 0, 139),  # dark red
    (0, 100, 0),  # dark green
    (139, 0, 0),  # dark blue
    (0, 140, 140),  # dark yellow/cyan-like
    (139, 0, 139),  # dark magenta
    (139, 139, 0),  # dark cyan
    (0, 69, 139),  # dark orange
    (75, 0, 130),  # indigo
    (47, 79, 79),  # dark slate gray
    (85, 107, 47),  # dark olive green
    (128, 0, 0),  # navy
    (0, 128, 128),  # teal
    (72, 61, 139),  # dark slate blue
    (34, 139, 34),  # forest green
    (25, 25, 112),  # midnight blue
]

def draw_trail(output_image_frame, object_trails, trail_length=500):
    trail_points = list(object_trails.values())
    trail_ids = list(object_trails.keys())
    for i in range(len(trail_points)):
        if len(trail_points[i]) > 1:
            for j in range(1, len(trail_points[i])):
                cv2.line(output_image_frame, (int(trail_points[i][j-1][0]), int(trail_points[i][j-1][1])),
                         (int(trail_points[i][j][0]), int(trail_points[i][j][1])), DARK_COLORS[trail_ids[i] % len(DARK_COLORS)], thickness=3)
        if len(trail_points[i]) > trail_length:
            trail_points[i].pop(0)  # Remove the oldest point from the trail

def _get_runtime_cfg():
    return cfg.get("BotSORT", {})

def start(show_window=True):
    runtime_cfg = _get_runtime_cfg()
    video_path = runtime_cfg.get("VIDEO_PATH")
    result_path = runtime_cfg.get("RESULT_PATH")
    max_frame = runtime_cfg.get("MAX_FRAME")

    # Initialize video capture to get video properties
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties (width and height)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Close the video capture
    capture.release()

    capture = cv2.VideoCapture(video_path)
    videoWriter = None
    input_video_fps = int(capture.get(cv2.CAP_PROP_FPS))
    print('fps:', input_video_fps)

    # Dictionary to store the trail points of each object
    object_trails = {}

    v11Tracker = yolov11Tracker()

    start_time = datetime.now()
    frames_processed = 0
    global_frame_id = 0
    out_filename = f"output_{start_time.strftime('%Y%m%d_%H%M%S')}_{Path(result_path).stem}.txt"
    out_filepath = os.path.join(os.getcwd(), out_filename)

    try:
        while True:
            if max_frame is not None and global_frame_id > max_frame:
                break
            global_frame_id += 1

            _, im = capture.read()
            if im is None:
                break

            detections = Detections()
            output_image_frame, list_bboxs = v11Tracker.track(im)
            frames_processed += 1

            for item_bbox in list_bboxs:
                x1, y1, x2, y2, class_label, track_id, confidence = item_bbox
                detections.add((x1, y1, x2, y2), None, None, track_id)

            # Add the current object's position to the trail
            for xyxy, _, _, track_id in detections.detections:
                x1, y1, x2, y2 = xyxy
                center = Point(x=(x1+x2)/2, y=(y1+y2)/2)
                if track_id in object_trails:
                    object_trails[track_id].append((center.x, center.y))
                else:
                    object_trails[track_id] = [(center.x, center.y)]

            # Draw the trail for each object
            # trail_colors = [(255, 0, 255)] * len(object_trails)  # Red color for all trails
            # draw_trail(output_image_frame, list(object_trails.values()))
            draw_trail(output_image_frame, object_trails)

            # Remove trails of objects that are not detected in the current frame
            for tracker_id in list(object_trails.keys()):
                if tracker_id not in [item[3] for item in detections.detections]:
                    object_trails.pop(tracker_id)

            if videoWriter is None:
                fourcc = cv2.VideoWriter_fourcc(
                    'm', 'p', '4', 'v')  # opencv3.0
                videoWriter = cv2.VideoWriter(
                    result_path, fourcc, input_video_fps, (output_image_frame.shape[1], output_image_frame.shape[0]))

            videoWriter.write(output_image_frame)
            if show_window:
                height, width = output_image_frame.shape[:2]
                output_image_frame = cv2.resize(output_image_frame, (int(width/3), int(height/3)))
                cv2.imshow('Demo', output_image_frame)
                cv2.waitKey(1)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        processing_fps = frames_processed / elapsed if elapsed > 0 else 0.0

        try:
            v11Tracker.save_csv(force=True)
        except Exception as e:
            print(f"Failed to flush tracking CSV: {e}")

        try:
            with open(out_filepath, 'w', encoding='utf-8') as f:
                f.write(f"start_time: {start_time.isoformat()}\n")
                f.write(f"end_time: {end_time.isoformat()}\n")
                f.write(f"elapsed_seconds: {elapsed:.3f}\n")
                f.write(f"input_video_fps: {input_video_fps}\n")
                f.write(f"processing_fps: {processing_fps:.3f}\n")
                f.write(f"frames_processed: {frames_processed}\n")
                f.write(f"VIDEO_PATH: {video_path}\n")
                f.write(f"RESULT_PATH: {result_path}\n")
                f.write(f"TRACKING_CSV_PATH: {v11Tracker.tracking_csv_path}\n")
                f.write(f"bot_sort_cfg: {cfg}\n")
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
    cfg_path = "config/hara_botsort.yaml"
    cfg.merge_from_file(cfg_path)
    cfg.setdefault("cfg_path", cfg_path)
    start()


    # #
    # # # HBB10
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min\hold1_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min\hold2_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min\hold3_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min\hold4_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min\hold5_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min\hold6_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    #
    # # Detection
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection\hold1_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection\hold2_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection\hold3_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection\hold4_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection\hold5_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection\hold6_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # #
    # #
    # # # Track
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_track\hold1_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_track\hold2_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_track\hold3_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_track\hold4_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_track\hold5_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_track\hold6_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)


    # # # detection track
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection_track\hold1_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection_track\hold2_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection_track\hold3_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection_track\hold4_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection_track\hold5_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)
    # cfg_path = r"C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\hara_report\large_model\HBB10min_detection_track\hold6_botsort.yaml"
    # cfg.merge_from_file(cfg_path)
    # cfg.setdefault("cfg_path", cfg_path)
    # start(show_window=False)








