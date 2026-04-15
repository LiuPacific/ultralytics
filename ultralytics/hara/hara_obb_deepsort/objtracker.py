from deep_sort.utils.parser import get_config
from utils.deep_sort_obb import DeepSortOBB
import torch
import cv2
import numpy as np

cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSortOBB(cfg.DEEPSORT.REID_CKPT,
                      max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                      nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                      max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                      use_cuda=True, use_rotated_features=True)


def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    list_pts = []
    point_radius = 4
    
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        if cls_id in ['smoke', 'phone', 'eat']:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        if cls_id == 'eat':
            cls_id = 'eat-drink'
            
        # check whether hit line 
        check_point_x = x1
        check_point_y = int(y1 + ((y2 - y1) * 0.6))

        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        list_pts.clear()            
        list_pts.append([check_point_x-point_radius, check_point_y-point_radius])
        list_pts.append([check_point_x-point_radius, check_point_y+point_radius])
        list_pts.append([check_point_x+point_radius, check_point_y+point_radius])
        list_pts.append([check_point_x+point_radius, check_point_y-point_radius])

        ndarray_pts = np.array(list_pts, np.int32)
        cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))
        #list_pts.clear()
    return image

def update(target_detector, image):
    detections = target_detector.detect(image)
    obb_list = []
    confs = []
    bboxes2draw = []

    if len(detections):
        for detection in detections:
            # detection is (bbox, label, confidence) where bbox is (4, 2) corner points
            bbox_corners, label, conf = detection
            obb_list.append(bbox_corners)
            confs.append(conf)

        # Pass OBB detections to deepsort
        outputs = deepsort.update(obb_list, confs, image)
        for value in list(outputs):
            x1, y1, x2, y2, track_id = value
            bboxes2draw.append((x1, y1, x2, y2, '', track_id))

    plot_all_obb_detections(image, detections)
    image = plot_bboxes(image, bboxes2draw)
    return image, bboxes2draw

def plot_all_obb_detections(image, detections, line_thickness=None):
    # Plots OBB detections on image
    tl = 5  # line/font thickness
    color = (0, 128, 128)

    for detection in detections:
        bbox_corners, label, conf = detection
        # Draw OBB as polygon
        pts = np.array(bbox_corners, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, color, thickness=tl, lineType=cv2.LINE_AA)

        # Add confidence text
        # Use centroid as text position
        center_x = int(np.mean(bbox_corners[:, 0]))
        center_y = int(np.mean(bbox_corners[:, 1]))
        tf = max(tl - 1, 1)  # font thickness
        cv2.putText(image, f'{label} {conf:.2f}', (center_x, center_y), 0, 1,
                    [225, 255, 0], thickness=tf, lineType=cv2.LINE_AA)

    return image