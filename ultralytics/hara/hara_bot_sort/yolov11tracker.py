import torch
import numpy as np
import cv2
import csv
import os
from ultralytics import YOLO
from ultralytics.hara.hara_bot_sort.config.common_cfg import cfg


OBJ_LIST = ['chicken']
DETECTOR_PATH = r'D:\chicken_project\experiment6report\running\hbb_l_hold1_10min\weights\last.pt'




class baseTracker(object):
    def __init__(self):
        self.img_size = 1280
        self.conf = 0.25
        self.iou = 0.70

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")



class yolov11Tracker(baseTracker):
    def __init__(self):
        super(yolov11Tracker, self).__init__()
        self.runtime_cfg = cfg.get("BotSORT", {}) or {}
        self.tracking_csv_path = self.runtime_cfg.get("TRACKING_CSV_PATH", "tracking_output1.csv")
        self._csv_buffer = []
        self._frames_since_flush = 0
        self._flush_interval = self.runtime_cfg.get("TRACKING_FLUSH_INTERVAL", 30)
        self.frame_id = 0
        self.init_model()

    def init_model(self):
        self.weights = self.runtime_cfg.get("DETECTION_MODEL_PATH", DETECTOR_PATH)
        self.img_size = self.runtime_cfg.get("IMG_SIZE", self.img_size)
        self.conf = self.runtime_cfg.get("CONF", self.conf)
        self.iou = self.runtime_cfg.get("NMS_THRESHOLD", 0.7)
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.weights)
        self.m = self.model
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def track(self, im):
        self.frame_id += 1
        res = self.model.track(im,
                               # tracker="bot_sort/botsort.yaml",
                               tracker=cfg.get("cfg_path"),
                               persist=True, imgsz=self.img_size, conf=self.conf,
                               iou=self.iou, device=self.device)

        detected_boxes = res[0].boxes
        pred_boxes = []
        for box in detected_boxes:
            if box.id is None:
                continue
            xyxy = box.xyxy.cpu()
            #print(xyxy)
            confidence = box.conf.cpu().item()
            class_id = box.cls  # get the class id
            class_id_cpu = class_id.cpu()  # move the value to CPU
            class_id_int = int(class_id_cpu.item())  # convert to integer
            lbl = self.names[class_id_int]
            if not lbl in OBJ_LIST:
                continue
            x1, y1, x2, y2 = xyxy[0].numpy()
            track_id = int(box.id.cpu().item())
            #print(x1, y1, x2, y2, lbl, confidence, track_id)
            pred_boxes.append(
                (x1, y1, x2, y2, '', track_id, confidence))
        self._save_tracking_information(pred_boxes)
        im = plot_bboxes(im, pred_boxes)
        # im = draw_trail(im, tracks2draw)
        return im, pred_boxes

    def _save_tracking_information(self, pred_boxes):
        """Record tracking information for current frame and flush to CSV when needed."""
        self.record_frame(self.frame_id, pred_boxes)
        self._frames_since_flush += 1
        if self._frames_since_flush >= self._flush_interval:
            self._frames_since_flush = 0
            self.save_csv()

    def record_frame(self, global_frame_id, pred_boxes):
        for x1, y1, x2, y2, _, track_id, confidence in pred_boxes:
            width = float(x2 - x1)
            height = float(y2 - y1)
            center_x = float(x1 + width / 2.0)
            center_y = float(y1 + height / 2.0)

            self._csv_buffer.append(
                (
                    int(global_frame_id),
                    int(track_id),
                    center_x,
                    center_y,
                    width,
                    height,
                    float(confidence) if confidence is not None else None,
                    True,
                )
            )

    def save_csv(self, file_path=None, force=False):
        if not self._csv_buffer and not force:
            return

        out_path = file_path or self.tracking_csv_path
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        file_exists = os.path.exists(out_path)
        with open(out_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['frame_id', 'track_id', 'center_x', 'center_y', 'width', 'height', 'confidence', 'detected'])
            for row in self._csv_buffer:
                writer.writerow([r if r is not None else '' for r in row])

        self._csv_buffer = []






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


def draw_trail(image, tracks2draw, trail_length=630):
    for track2draw in tracks2draw:
        track_history_positions = track2draw[6]
        track_id = track2draw[5]

        if len(track_history_positions) == 0:
            return image

        # use only recent trail points.
        if len(track_history_positions) > trail_length:
            track_history_positions = track_history_positions[-trail_length:]
        # assign a different color to each track:
        for i in range(1, len(track_history_positions)):
            pt1 = (int(track_history_positions[i - 1][0]), int(track_history_positions[i - 1][1]))
            pt2 = (int(track_history_positions[i][0]), int(track_history_positions[i][1]))
            cv2.line(image, pt1, pt2, DARK_COLORS[track_id % len(DARK_COLORS)], thickness=3)

    return image


def plot_all_detections(image, detected_bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = 2  # line/font thickness
    color = (0, 0, 128)

    for x1, y1, x2, y2, _, conf in detected_bboxes:
        c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        cv2.putText(image, '{}'.format(round(conf.item(),2)), (c1[0], c1[1] + 20), 0, 1,
                    [0, 255, 0], thickness=tf, lineType=cv2.LINE_AA)

    return image



def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness

    color = (0, 0, 256)
    for (x1, y1, x2, y2, _, track_id, _) in bboxes:
        c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        center_x = int((x1+x2)/2)
        center_y = int((y1+y2)/2)
        cv2.putText(image, f'{int(track_id)}', (center_x, center_y), 0, 1,
                    [0, 255, 0], thickness=tf, lineType=cv2.LINE_AA)
    return image


