import numpy as np
import torch
import sys
import os

# Add parent directory to path for imports
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .tracker_obb import TrackerOBB
from .detection_obb import OBBDetection
from ultralytics.hara.hara_obb_deepsort.deep_sort.deep_sort.deep.feature_extractor import Extractor
from ultralytics.hara.hara_obb_deepsort.deep_sort.deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from .kalman_filter_obb import KalmanFilterOBB
from .obb_utils import extract_obb_crop, xyxyxyxy_to_xywhr


class DeepSortOBB(object):
    """
    Extended DeepSort for oriented bounding box (OBB) tracking.
    """

    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0,
                 max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True,
                 use_rotated_features=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.use_rotated_features = use_rotated_features

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

        # Use OBB Kalman filter instead of standard one
        self.kalman_filter = KalmanFilterOBB()
        self.tracker = TrackerOBB(metric, max_iou_distance=max_iou_distance, max_age=max_age,
                                  n_init=n_init, kalman_filter=self.kalman_filter)

    def update(self, obb_detections, confidences, ori_img):
        """
        Update tracker with OBB detections.

        Parameters
        ----------
        obb_detections : list of ndarray
            List of OBB corner points, each as (4, 2) array
        confidences : list
            Detection confidence scores
        ori_img : ndarray
            Original image for feature extraction

        Returns
        -------
        ndarray
            Tracked objects as [x1,y1,x2,y2,track_id] or OBB format
        """
        self.height, self.width = ori_img.shape[:2]

        # Extract features from OBB crops
        features = self._get_features_obb(obb_detections, ori_img)

        # Create OBB detection objects
        detections = [OBBDetection(obb_detections[i], conf, features[i])
                      for i, conf in enumerate(confidences) if conf > self.min_confidence]

        # Update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # Output tracked objects
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # Get track state and convert to output format
            # For now, return axis-aligned approximation
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=int))

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def update_xywhr(self, obb_xywhr, confidences, ori_img):
        """
        Update tracker with xywhr format OBB detections.

        Parameters
        ----------
        obb_xywhr : ndarray
            OBB detections in (cx, cy, w, h, angle) format, shape (N, 5)
        confidences : list
            Detection confidence scores
        ori_img : ndarray
            Original image for feature extraction

        Returns
        -------
        ndarray
            Tracked objects
        """
        # Convert xywhr to xyxyxyxy format for feature extraction
        from obb_utils import xywhr_to_xyxyxyxy
        obb_corners = []
        for xywhr in obb_xywhr:
            corners = xywhr_to_xyxyxyxy(*xywhr)
            obb_corners.append(corners)

        return self.update(obb_corners, confidences, ori_img)

    def _get_features_obb(self, obb_detections, ori_img):
        """
        Extract features from OBB crops.

        Parameters
        ----------
        obb_detections : list
            List of OBB corner points
        ori_img : ndarray
            Original image

        Returns
        -------
        list
            List of feature vectors
        """
        im_crops = []

        for obb_corners in obb_detections:
            if self.use_rotated_features:
                # Extract rotated crop aligned to OBB orientation
                try:
                    crop = extract_obb_crop(ori_img, obb_corners, padding=10)
                    im_crops.append(crop)
                except Exception as e:
                    print(f"Warning: Failed to extract rotated crop, using axis-aligned fallback: {e}")
                    # Fallback to axis-aligned crop
                    x_coords = obb_corners[:, 0]
                    y_coords = obb_corners[:, 1]
                    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(ori_img.shape[1], x_max)
                    y_max = min(ori_img.shape[0], y_max)

                    crop = ori_img[y_min:y_max, x_min:x_max]
                    im_crops.append(crop)
            else:
                # Use axis-aligned crop
                x_coords = obb_corners[:, 0]
                y_coords = obb_corners[:, 1]
                x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(ori_img.shape[1], x_max)
                y_max = min(ori_img.shape[0], y_max)

                crop = ori_img[y_min:y_max, x_min:x_max]
                im_crops.append(crop)

        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])

        return features

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """Convert tlwh to xyxy format."""
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2
