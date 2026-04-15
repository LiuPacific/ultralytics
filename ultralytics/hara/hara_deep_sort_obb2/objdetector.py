import torch
from ultralytics import YOLO
import numpy as np

# OBJ_LIST = ['person', 'car', 'bus', 'truck']
# DETECTOR_PATH = r'G:\project_chicken\code\experiment_deepSORT\weights\yolov8s.pt'
OBJ_LIST = ['chicken', 'bus']
DETECTOR_PATH = r'E:\chicken_project\workspace\ultralytics\ultralytics\hara\hara_deep_sort\.weights\best.pt'

class baseDet(object):
    def __init__(self):
        self.img_size = 1024  # Increased for OBB detection
        self.conf = 0.25
        self.iou = 0.70

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

    def init_model(self):
        self.weights = DETECTOR_PATH
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.weights)
        self.m = self.model
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def detect(self, im):
        res = self.model.predict(im, imgsz=self.img_size, conf=self.conf,
                                     iou=self.iou, device=self.device)

        detections = []

        # Try OBB detection first
        if hasattr(res[0], 'obb') and res[0].obb is not None:
            # Use OBB detection
            obb_xyxyxyxy = res[0].obb.xyxyxyxy.cpu().numpy()   # shape: (N, 4, 2)
            obb_conf = res[0].obb.conf.cpu().numpy()           # shape: (N,)
            obb_cls = res[0].obb.cls.cpu().numpy().astype(int) # shape: (N,)

            for pts, score, cls_id in zip(obb_xyxyxyxy, obb_conf, obb_cls):
                bbox = np.array(pts, dtype=np.int32).reshape((-1,1,2))
                lbl = self.names[cls_id]
                if not lbl in OBJ_LIST:
                    continue
                detections.append((bbox, lbl, score))
        else:
            # Fallback to regular bounding boxes
            print("Warning: OBB not available, falling back to axis-aligned boxes")
            detected_boxes = res[0].boxes
            for box in detected_boxes:
                xyxy = box.xyxy.cpu()
                confidence = box.conf.cpu()
                class_id = box.cls.cpu().int()
                lbl = self.names[class_id.item()]
                if not lbl in OBJ_LIST:
                    continue
                x1, y1, x2, y2 = xyxy[0].numpy()
                # Convert axis-aligned box to OBB format (rectangle)
                bbox = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32).reshape((-1,1,2))
                detections.append((bbox, lbl, confidence.item()))

        return detections
