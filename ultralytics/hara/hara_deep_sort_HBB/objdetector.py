import torch
from ultralytics import YOLO
from ultralytics.hara.hara_deep_sort_HBB.deep_sort.configs.common_cfg import cfg

# OBJ_LIST = ['person', 'car', 'bus', 'truck']
# DETECTOR_PATH = r'G:\project_chicken\code\experiment_deepSORT\weights\yolov8s.pt'
OBJ_LIST = ['chicken']

class baseDet(object):
    def __init__(self):
        self.img_size = 1280
        self.conf = 0.25
        self.iou = 0.80

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
        self.weights = cfg.DEEPSORT.DETECTION_MODEL_PATH
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


    def detect(self, im):
        res = self.model.predict(im, imgsz=self.img_size, conf=self.conf,
                                     iou=self.iou, device=self.device)
                    
        detected_boxes = res[0].boxes
        pred_boxes = []
        for box in detected_boxes:
            xyxy = box.xyxy.cpu() 
            #print(xyxy)
            confidence = box.conf.cpu() 
            class_id = box.cls  # get the class id
            class_id_cpu = class_id.cpu()  # move the value to CPU
            class_id_int = int(class_id_cpu.item())  # convert to integer
            lbl = self.names[class_id_int]
            if not lbl in OBJ_LIST:
                continue
            x1, y1, x2, y2 = xyxy[0].numpy()
            pred_boxes.append(
                 (x1, y1, x2, y2, lbl, confidence))
        return im, pred_boxes

