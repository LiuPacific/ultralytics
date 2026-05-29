import torch
import numpy as np
import cv2
from ultralytics import YOLO

OBJ_LIST = ['person', 'car', 'bus', 'truck']
# DETECTOR_PATH = '.weights/yolov8s.pt'
# OBJ_LIST = ['chicken', 'bus']
DETECTOR_PATH = r'C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolo11l.pt'
# DETECTOR_PATH = r'C:\Users\tliu25\workspace\ultralytics\ultralytics\hara\weights\yolov8s.pt'


class baseTracker(object):
    def __init__(self):
        self.img_size = 640
        self.conf = 0.25
        self.iou = 0.70

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")


def draw_bboxes(im, pred_boxes):
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]  # Green, Blue, Red, Yellow colors for different classes
    magenta_color = (255, 0, 255)  # Magenta color for text label
    for box in pred_boxes:
        x1, y1, x2, y2, lbl, _, track_id = box
        class_idx = OBJ_LIST.index(lbl) if lbl in OBJ_LIST else -1
        if class_idx != -1:
            color = colors[class_idx]
        else:
            color = (0, 0, 0)  # Default color if class not in OBJ_LIST
        thickness = 2

        # Draw the bounding box
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        # Add text label with track_id in magenta color
        text = f'{lbl} (ID:{track_id})'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.rectangle(im, (int(x1), int(y1 - text_size[1] - 5)),
                      (int(x1 + text_size[0]), int(y1 - 5)), color, -1)
        cv2.putText(im, text, (int(x1), int(y1 - 5)),
                    font, font_scale, magenta_color, font_thickness, lineType=cv2.LINE_AA)

    return im

class yolov11Tracker(baseTracker):
    def __init__(self):
        super(yolov11Tracker, self).__init__()
        self.init_model()

    def init_model(self):
        self.weights = DETECTOR_PATH
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.weights)
        self.m = self.model
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def track(self, im):
        res = self.model.track(im, tracker="botsort.yaml", persist=True, imgsz=self.img_size, conf=self.conf,
                               iou=self.iou, device=self.device)

        detected_boxes = res[0].boxes
        pred_boxes = []
        for box in detected_boxes:
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
                (x1, y1, x2, y2, lbl, confidence, track_id))
        im = draw_bboxes(im, pred_boxes)
        return im, pred_boxes

