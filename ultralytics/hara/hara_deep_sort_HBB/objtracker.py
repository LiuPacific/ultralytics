from deep_sort.deep_sort import DeepSort
import torch
import cv2
import numpy as np



def update(target_detector, image, deepsort: DeepSort):
    _, detected_bboxes, pred_boxes = target_detector.detect(image)
    bbox_xywh = []
    confs = []
    tracks2draw = []

    if len(detected_bboxes):
        # Adapt detections to deep sort input format
        for x1, y1, x2, y2, _, conf in detected_bboxes:
            obj = [
                int((x1+x2)/2), int((y1+y2)/2),
                x2-x1, y2-y1
            ]
            bbox_xywh.append(obj)
            confs.append(conf)
        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)

        # Pass detections to deepsort
        tracks_detected = deepsort.update(xywhs, confss, image)
        for track_detected in list(tracks_detected):
            x1,y1,x2,y2,track_id, track_history_positions = track_detected
            tracks2draw.append(
                (x1, y1, x2, y2, '', track_id,track_history_positions)
            )

    image = plot_all_detections(image, pred_boxes)
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
    list_pts = []

    color = (0, 0, 256)
    for (x1, y1, x2, y2, _, track_id, _) in bboxes:
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        center_x = int((x1+x2)/2)
        center_y = int((y1+y2)/2)
        cv2.putText(image, f'{int(track_id)}', (center_x, center_y), 0, 1,
                    [0, 255, 0], thickness=tf, lineType=cv2.LINE_AA)
    return image