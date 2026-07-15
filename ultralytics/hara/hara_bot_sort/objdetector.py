import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.hara.hara_bot_sort.config.common_cfg import cfg


OBJ_LIST = ['chicken']

class baseDet(object):
    def __init__(self):
        self.img_size = 1280
        self.conf = 0.25
        self.nms_iou = cfg.DEEPSORT.get("NMS_THRESHOLD", 0.7)
        # self.iou = 0.80


    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")

class Detector( baseDet):
    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.bbox_history = []
        self.frame_memory_length = 5

    def init_model(self):
        self.weights = cfg.get("BotSORT").get("DETECTION_MODEL_PATH")
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.weights)
        self.m = self.model
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def _filter_by_area(self, xywhr, min_area=20000, max_area=120000):
        """Filter bbox based on area constraints."""
        # area = self._calculate_bbox_area(xyxyxyxy)
        area = xywhr[2] * xywhr[3]
        if min_area <= area <= max_area:
            return True
        return False

    def _update_frame_memory(self, pred_boxes):
        """Update the frame history with new detections."""
        # Remove old frames that exceed memory length
        if len(self.bbox_history) >= self.frame_memory_length:
            self.bbox_history.pop(0)
        self.bbox_history.append(pred_boxes)

    def _bbox_center(self, box):
        x1, y1, x2, y2 = box[:4]
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]