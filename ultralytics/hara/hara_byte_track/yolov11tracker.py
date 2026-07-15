import csv
import os

import cv2
import torch
from ultralytics import YOLO

from ultralytics.hara.hara_byte_track.config.common_cfg import cfg


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
        self.runtime_cfg = cfg.get("ByteTrack", {})
        self.tracking_csv_path = self.runtime_cfg.get("TRACKING_CSV_PATH") or "tracking_output.csv"
        self._csv_buffer = []
        self._frames_since_flush = 0
        self._flush_interval = self.runtime_cfg.get("TRACKING_FLUSH_INTERVAL", 30)
        self.frame_id = 0
        self.init_model()

    def init_model(self):
        self.weights = self.runtime_cfg.get("DETECTION_MODEL_PATH", DETECTOR_PATH)
        self.img_size = self.runtime_cfg.get("IMG_SIZE", self.img_size)
        self.conf = self.runtime_cfg.get("CONF", self.conf)
        self.iou = self.runtime_cfg.get("IOU", self.iou)
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.weights)
        self.m = self.model
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def track(self, im):
        self.frame_id += 1
        res = self.model.track(
            im,
            # tracker="byte_track/hara_bytetrack.yaml",
            tracker=cfg.get("cfg_path"),
            persist=True,
            imgsz=self.img_size,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
        )

        detected_boxes = res[0].boxes
        pred_boxes = []
        for box in detected_boxes:
            if box.id is None:
                continue
            xyxy = box.xyxy.cpu()
            confidence = box.conf.cpu().item()
            class_id = box.cls
            class_id_cpu = class_id.cpu()
            class_id_int = int(class_id_cpu.item())
            lbl = self.names[class_id_int]
            if lbl not in OBJ_LIST:
                continue
            x1, y1, x2, y2 = xyxy[0].numpy()
            track_id = int(box.id.cpu().item())
            pred_boxes.append((x1, y1, x2, y2, '', track_id, confidence))

        self._save_tracking_information(pred_boxes)
        im = plot_bboxes(im, pred_boxes)
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


def plot_bboxes(image, bboxes, line_thickness=None):
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1

    color = (0, 0, 256)
    for (x1, y1, x2, y2, _, track_id, _) in bboxes:
        c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.putText(image, f'{int(track_id)}', (center_x, center_y), 0, 1,
                    [0, 255, 0], thickness=tf, lineType=cv2.LINE_AA)
    return image
