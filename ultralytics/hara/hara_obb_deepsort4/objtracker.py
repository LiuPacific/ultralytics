
import torch
import cv2
import numpy as np

from ultralytics.hara.hara_obb_deepsort4.deep_sort.deep_sort.deep_sort_obb import DeepSORTOBB
from ultralytics.hara.hara_obb_deepsort4.obj_obb_detector import ObbDetector
from deep_sort.deep_sort.sort import obb_utils
from ultralytics.hara.hara_obb_deepsort4.deep_sort.configs.common_cfg import cfg

# deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
#                     max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
#                     nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
#                     max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
#                     use_cuda=True)



deepsort_obb = DeepSORTOBB(cfg.DEEPSORT.REID_CKPT,
                           max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                           nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                           max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                           use_cuda=True,
                           MAX_ID_POOL=cfg.DEEPSORT.MAX_ID_POOL,
                           reconnection_distance_threshold=cfg.DEEPSORT.RECONNECTION_DISTANCE_THRESHOLD,
                           reuse_id_assignment_distance_threshold=cfg.DEEPSORT.REUSE_ID_ASSIGNMENT_DISTANCE_THRESHOLD,
                           use_reid=cfg.DEEPSORT.USE_REID,
                           use_rotated_features=cfg.DEEPSORT.USE_ROTATED_FEATURES
                           )


def update(target_detector: ObbDetector, image):
    obb_detections = target_detector.detect(image)
    xyxyxyxy_list = []
    xywhr_list = []
    conf_list = []
    tracks2draw = []
    image = plot_all_detections(image, obb_detections)
    if len(obb_detections):
        # Adapt detections to deep sort input format
        for detection in obb_detections:
            xyxyxyxy, xywhr, label, conf = detection
            xyxyxyxy_list.append(xyxyxyxy)
            xywhr_list.append(xywhr)
            conf_list.append(conf)

        # Pass detections to deepsort
        tracks_detected = deepsort_obb.update(xyxyxyxy_list, xywhr_list, conf_list, image)

        for track_detected in list(tracks_detected):
            """
            track_detected: x y w h r track_id, [[xywhr],[xywhr]...[new xywhr]]"""
            x, y, w, h, r, track_id, track_history_positions = track_detected
            xyxyxyxy = obb_utils.xywhr_to_xyxyxyxy(x, y, w, h, r)
            tracks2draw.append(
                (xyxyxyxy, '', track_id, track_history_positions)  # xyxyxyxy, class_id, track_id
            )

    image = draw_trail(image, tracks2draw)
    image = plot_bboxes(image, tracks2draw)
    return image, tracks2draw


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
        track_history_positions = track2draw[3]
        track_id = track2draw[2]

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


def plot_all_detections(image, obb_detections, line_thickness=None):
    # Plots one bounding box on image img
    tl = 2  # line/font thickness
    color = (0, 0, 128)

    # for x1, y1, x2, y2, _, conf in detected_bboxes:
    #     c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
    #     cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    #     tf = max(tl - 1, 1)  # font thickness
    #     cv2.putText(image, '{}'.format(round(conf.item(),2)), (c1[0], c1[1] + 20), 0, 1,
    #                 [225, 255, 0], thickness=tf, lineType=cv2.LINE_AA)

    for detection in obb_detections:
        # xyxyxyxy (N, 4, 2)
        xyxyxyxy, xywhr, label, conf = detection
        # Draw OBB as polygon
        pts = np.array(xyxyxyxy, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, color, thickness=tl, lineType=cv2.LINE_AA)

        # Add confidence text
        # Use centroid as text position
        center_x = int(np.mean(xyxyxyxy[:, 0]))
        center_y = int(np.mean(xyxyxyxy[:, 1]))
        tf = max(tl - 1, 1)  # font thickness
        # cv2.putText(image, f'{label} {conf:.2f}', (center_x, center_y), 0, 1,
        #             [0, 255, 0], thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(image, f'    {conf:.2f}', (center_x, center_y), 0, 1,
                    [0, 255, 0], thickness=tf, lineType=cv2.LINE_AA)

    return image


# plot bboxes that are in the tracker
def plot_bboxes(image, bboxes2draw, line_thickness=None):
    """
    :param image:
    :param bboxes2draw:  xyxyxyxy, class_id, track_id
    :param line_thickness:
    :return:
    """
    tl = 5  # line/font thickness
    color = (0, 0, 256)
    for (xyxyxyxy, _, track_id, _) in bboxes2draw:
        # Draw OBB as polygon
        pts = np.array(xyxyxyxy, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, color, thickness=tl, lineType=cv2.LINE_AA)

        # Add confidence text
        # Use centroid as text position
        center_x = int(np.mean(xyxyxyxy[:, 0]))
        center_y = int(np.mean(xyxyxyxy[:, 1]))
        tf = max(tl - 1, 1)  # font thickness
        cv2.putText(image, f'{int(track_id)}', (center_x, center_y), 0, 1,
                    [0, 255, 0], thickness=tf, lineType=cv2.LINE_AA)

    return image

# plot bbox with line restriction check. When the detection is out of the line, it will show different color.
# def plot_bboxes(image, bboxes, line_thickness=None):
#     # Plots one bounding box on image img
#     tl = line_thickness or round(
#         0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
#     list_pts = []
#     point_radius = 4
#
#     for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
#         if cls_id in ['smoke', 'phone', 'eat']:
#             color = (0, 0, 255)
#         else:
#             color = (0, 255, 0)
#         if cls_id == 'eat':
#             cls_id = 'eat-drink'
#
#         # check whether hit line
#         check_point_x = x1
#         check_point_y = int(y1 + ((y2 - y1) * 0.6))
#
#         c1, c2 = (x1, y1), (x2, y2)
#         cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#         tf = max(tl - 1, 1)  # font thickness
#         t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
#         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
#         cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
#         cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
#                     [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
#         list_pts.clear()
#         list_pts.append([check_point_x - point_radius, check_point_y - point_radius])
#         list_pts.append([check_point_x - point_radius, check_point_y + point_radius])
#         list_pts.append([check_point_x + point_radius, check_point_y + point_radius])
#         list_pts.append([check_point_x + point_radius, check_point_y - point_radius])
#
#         ndarray_pts = np.array(list_pts, np.int32)
#         cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))
#         # list_pts.clear()
#     return image
