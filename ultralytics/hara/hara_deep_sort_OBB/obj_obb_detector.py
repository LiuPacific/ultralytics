import torch
from sympy import false

from ultralytics import YOLO
import numpy as np
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics.hara.hara_deep_sort_OBB.deep_sort.configs.common_cfg import cfg

# OBJ_LIST = ['person', 'chicken']
OBJ_LIST = ['chicken']


class baseDet(object):
    def __init__(self):
        self.img_size = 1280
        self.conf = 0.25
        # self.iou = 0.70
        self.nms_iou = cfg.DEEPSORT.get("NMS_THRESHOLD", 0.7)

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")


class ObbDetector(baseDet):
    DETECTION_COUNT_SAMPLE_INTERVAL = 15
    DETECTION_COUNT_STABILITY_WINDOW = 3

    def __init__(self):
        super(ObbDetector, self).__init__()
        self.init_model()
        # Frame memory for bbox tracking - stores bboxes for 3 frames
        self.bbox_history = []  # List of pred_boxes for the last 3 frames
        self.frame_memory_length = 3
        self.detection_frame_count = 0
        self.detection_count_history = []
        self.detection_optimization_target = None

    def init_model(self):
        self.weights = cfg.DEEPSORT.DETECTION_MODEL_PATH
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.weights)
        self.m = self.model
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    # def _calculate_bbox_area(self, xyxyxyxy):
    #     """Calculate the pixel area of a bbox from 4 corner points."""
    #     # xyxyxyxy has shape (4, 2) representing 4 corner points
    #     # Calculate area using the Shoelace formula
    #     x_coords = xyxyxyxy[:, 0]
    #     y_coords = xyxyxyxy[:, 1]
    #     area = 0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1)))
    #     return area

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

    def _update_detection_optimization_target(self, detection_count):
        """Sample the detection count and update the target after five stable samples."""
        self.detection_frame_count += 1
        if self.detection_frame_count % self.DETECTION_COUNT_SAMPLE_INTERVAL != 0:
            return

        self.detection_count_history.append(detection_count)
        if len(self.detection_count_history) > self.DETECTION_COUNT_STABILITY_WINDOW:
            self.detection_count_history.pop(0)

        if (
            len(self.detection_count_history) == self.DETECTION_COUNT_STABILITY_WINDOW
            and len(set(self.detection_count_history)) == 1
        ):
            new_target = self.detection_count_history[0]
            if new_target != self.detection_optimization_target:
                self.detection_optimization_target = new_target
                print(f"Detection optimization target updated to {new_target}.")

    @staticmethod
    def _blackout_outside_detection_region(im, x_min, x_max, y_min, y_max):
        """Return an image with pixels outside the inclusive detection region set to black."""
        if not isinstance(im, np.ndarray) or im.ndim < 2:
            raise ValueError("im must be a NumPy image with at least two dimensions.")

        height, width = im.shape[:2]
        x_start = max(0, min(int(x_min), width))
        x_stop = max(0, min(int(x_max) + 1, width))
        y_start = max(0, min(int(y_min), height))
        y_stop = max(0, min(int(y_max) + 1, height))

        masked_im = np.zeros_like(im)
        if x_start < x_stop and y_start < y_stop:
            masked_im[y_start:y_stop, x_start:x_stop] = im[y_start:y_stop, x_start:x_stop]
        return masked_im

    def detect(self, im, x_min=270, x_max=1900, y_min=100, y_max=1600):
        im = self._blackout_outside_detection_region(im, x_min, x_max, y_min, y_max)
        res = self.model.predict(im, imgsz=self.img_size, conf=self.conf,
                                 iou=self.nms_iou, device=self.device)

        obb_xyxyxyxy = res[0].obb.xyxyxyxy.cpu().numpy()  # shape: (N, 4, 2)
        obb_xywhr = res[0].obb.xywhr.cpu().numpy()  # shape: (N, 5)
        obb_conf = res[0].obb.conf.cpu().numpy()  # shape: (N,)
        obb_cls = res[0].obb.cls.cpu().numpy().astype(int)  # shape: (N,)
        pred_boxes = []
        for xyxyxyxy, xywhr, conf, cls_id in zip(obb_xyxyxyxy, obb_xywhr, obb_conf, obb_cls):
            # bbox = np.array(xyxyxyxy, dtype=np.int32).reshape((-1,1,2))
            xyxyxyxy = np.array(xyxyxyxy, dtype=np.int32)
            xywhr = np.array([int(xywhr[0]), int(xywhr[1]), int(xywhr[2]), int(xywhr[3]), xywhr[4]])
            lbl = self.names[cls_id]
            if not lbl in OBJ_LIST:
                continue

            # Filter by the pen boundary
            # if not (470 <= xywhr[0] <= 1900 and 670 <= xywhr[1] <= 1600):
            # if (xywhr[0] > x_max) or xywhr[0] < x_min or xywhr[1] > y_max or xywhr[1] < y_min:
            #     continue

            # Filter by bbox area (min: 13000, max: 120000 pixels)
            # if not self._filter_by_area(xywhr):
            #     continue
            pred_boxes.append(
                (xyxyxyxy, xywhr, lbl, conf)
            )

        selected_detections = pred_boxes
        if cfg.DEEPSORT.USE_OPTIMIZATION and cfg.DEEPSORT.get("DETECTION_OPTIMIZATION_ON", True):
            self._update_detection_optimization_target(len(pred_boxes))
            target_count = self.detection_optimization_target
            if (
                target_count is not None
                and len(pred_boxes) > target_count
                and len(self.bbox_history) > 0
                and len(self.bbox_history[-1]) == target_count
            ):
                prev_points = np.array(
                    [[box[1][0], box[1][1]] for box in self.bbox_history[-1]], dtype=float
                ).reshape(-1, 2)
                curr_points = np.array(
                    [[box[1][0], box[1][1]] for box in pred_boxes], dtype=float
                ).reshape(-1, 2)
                curr_scores = np.array([box[3] for box in pred_boxes])  # shape: (m,)
                _, _, _, selected_indices = select_points_by_distance_and_confidence(
                    prev_points, curr_points, curr_scores, alpha_distance=0.7, alpha_score=0.3,
                    x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
                )
                # Update pred_boxes to only include the selected points
                selected_detections = [pred_boxes[i] for i in selected_indices]
            self._update_frame_memory(selected_detections)

        # Update frame memory with current detections
        return selected_detections, pred_boxes


def select_points_by_distance_and_confidence(
        prev_points,
        curr_points,
        curr_scores,
        alpha_distance=0.7,
        alpha_score=0.3,
        x_min=470,
        x_max=1900,
        y_min=670,
        y_max=1600,
):
    """
    Select a dynamic number of points by matching them to the previous detections.

    Parameters
    ----------
    prev_points : np.ndarray
        Shape: (n, 2)
        Points from previous frame/image. Each row is [x, y].

    curr_points : np.ndarray
        Shape: (m, 2)
        Points from current frame/image. Each row is [x, y].

    curr_scores : np.ndarray
        Shape: (m,)
        Confidence scores of current points, values from 0 to 1.

    alpha_distance : float
        Weight for distance cost. Larger means movement smoothness is more important.

    alpha_score : float
        Weight for confidence reward. Larger means confidence score is more important.

    Returns
    -------
    selected_points : np.ndarray
        Shape: (n, 2), selected current-frame points.

    selected_scores : np.ndarray
        Shape: (n,), confidence scores of selected points.

    matched_pairs : list
        List of tuples: (prev_index, curr_index, distance, score, cost)

    curr_indices : list
        List of indices of selected points in the original curr_points array.
    """

    prev_points = np.asarray(prev_points, dtype=float)
    curr_points = np.asarray(curr_points, dtype=float)
    curr_scores = np.asarray(curr_scores, dtype=float)

    if prev_points.ndim != 2 or prev_points.shape[1] != 2:
        raise ValueError("prev_points must have shape (n, 2).")

    if curr_points.ndim != 2 or curr_points.shape[1] != 2:
        raise ValueError("curr_points must have shape (m, 2).")

    if curr_scores.shape[0] != curr_points.shape[0]:
        raise ValueError("curr_scores length must match curr_points length.")

    target_count = prev_points.shape[0]
    if curr_points.shape[0] < target_count:
        raise ValueError("curr_points must contain at least as many points as prev_points.")

    # Maximum possible distance in your valid image region
    d_max = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)

    # Distance matrix, shape: (n, m)
    # distances[i, j] = distance between previous point i and current point j
    diff = prev_points[:, None, :] - curr_points[None, :, :]
    distances = np.linalg.norm(diff, axis=2)

    # Normalize distance to roughly 0-1
    norm_distances = distances / d_max

    # Cost matrix:
    # lower distance is better, higher score is better
    cost_matrix = alpha_distance * norm_distances - alpha_score * curr_scores[None, :]

    # Hungarian algorithm
    prev_indices, curr_indices = linear_sum_assignment(cost_matrix)

    selected_points = curr_points[curr_indices]
    selected_scores = curr_scores[curr_indices]

    matched_pairs = []
    for i, j in zip(prev_indices, curr_indices):
        matched_pairs.append(
            {
                "prev_index": int(i),
                "curr_index": int(j),
                "distance": float(distances[i, j]),
                "score": float(curr_scores[j]),
                "cost": float(cost_matrix[i, j]),
            }
        )

    return selected_points, selected_scores, matched_pairs, curr_indices
