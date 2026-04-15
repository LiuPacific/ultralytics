import torch
from ultralytics import YOLO
import numpy as np

# OBJ_LIST = ['person', 'car', 'bus', 'truck']
# DETECTOR_PATH = r'G:\project_chicken\code\experiment_deepSORT\weights\yolov8s.pt'
OBJ_LIST = ['chicken']
DETECTOR_PATH = r'C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolov8m-obb-chicken.pt'

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


class ObbDetector(baseDet):
    def __init__(self):
        super(ObbDetector, self).__init__()
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
        obb_xywhr = res[0].obb.xywhr.cpu().numpy()           # shape: (N, 5)
        obb_conf = res[0].obb.conf.cpu().numpy()           # shape: (N,)
        obb_cls = res[0].obb.cls.cpu().numpy().astype(int) # shape: (N,)
        pred_boxes = []
        for xyxyxyxy, xywhr, conf, cls_id in zip(obb_xyxyxyxy,obb_xywhr, obb_conf,obb_cls):
            # bbox = np.array(xyxyxyxy, dtype=np.int32).reshape((-1,1,2))
            xyxyxyxy = np.array(xyxyxyxy, dtype=np.int32)
            xywhr = np.array(xywhr, dtype=np.int32)
            lbl = self.names[cls_id]
            if not lbl in OBJ_LIST:
                continue
            pred_boxes.append(
                (xyxyxyxy, xywhr, lbl, conf)
            )
        return pred_boxes


