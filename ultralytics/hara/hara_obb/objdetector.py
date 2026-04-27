import torch
from ultralytics import YOLO
import numpy as np

# OBJ_LIST = ['person', 'car', 'bus', 'truck']
# DETECTOR_PATH = r'G:\project_chicken\code\experiment_deepSORT\weights\yolov8s.pt'
OBJ_LIST = ['chicken']
DETECTOR_PATH = r'/ultralytics/hara/weights/yolov8m-obb-chicken-0401.pt'

class baseDet(object):
    def __init__(self):
        self.img_size = 1024
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

        obb_xyxyxyxy = res[0].obb.xyxyxyxy.cpu().numpy()   # shape: (N, 4, 2)
        obb_conf = res[0].obb.conf.cpu().numpy()           # shape: (N,)
        obb_cls = res[0].obb.cls.cpu().numpy().astype(int) # shape: (N,)
        detections = []
        for pts, score, cls_id in zip(obb_xyxyxyxy,obb_conf,obb_cls):
            bbox = np.array(pts, dtype=np.int32).reshape((-1,1,2))
            lbl = self.names[cls_id]
            if not lbl in OBJ_LIST:
                continue
            detections.append(
                (bbox, lbl, obb_conf)
            )
        return detections


