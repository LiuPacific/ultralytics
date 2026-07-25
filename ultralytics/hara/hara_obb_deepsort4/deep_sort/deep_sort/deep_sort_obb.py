import numpy as np
import torch
import math

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection_obb import DetectionOBB
from .sort.tracker_obb import TrackerOBB
from .sort.kalman_filter_obb import KalmanFilterOBB
from .sort import obb_utils

from ultralytics.hara.hara_reid import features_extractor


# __all__ = ['DeepSort'] # __all__ defines the public API exported by this module.

class DeepSORTOBB(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3,
                 max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True, MAX_ID_POOL=15,
                 use_rotated_features=True, reconnection_distance_threshold=400,
                 reuse_id_assignment_distance_threshold=200, use_reid=False, tracking_csv_path=None):
        self.min_confidence = min_confidence
        self.use_rotated_features = use_rotated_features
        self.use_reid = use_reid
        if use_reid:
            self.extractor = features_extractor.ChickenFeatureExtractor(model_path)
        # self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

        # Use OBB Kalman filter instead of standard one
        self.kalman_filter = KalmanFilterOBB()
        self.tracker_obb = TrackerOBB(metric, max_iou_distance=max_iou_distance, max_age=max_age,
                                      n_init=n_init, kalman_filter=self.kalman_filter, MAX_ID_POOL=MAX_ID_POOL,
                                      reconnection_distance_threshold=reconnection_distance_threshold,
                                      reuse_id_assignment_distance_threshold=reuse_id_assignment_distance_threshold,
                                      tracking_csv_path=tracking_csv_path
                                      )

    def update(self, xyxyxyxy_list, xywhr_list, confidences, ori_img):
        """
        Update tracker with OBB detections.

        Parameters
        ----------
        xywhr_list :
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
        detections = []
        if self.use_reid:
            features = self._get_features_obb(xyxyxyxy_list, ori_img)
            # Create OBB detection objects
            detections = [DetectionOBB(xyxyxyxy_list[i], xywhr_list[i], conf, features[i])
                          for i, conf in enumerate(confidences) if conf > self.min_confidence]
        else:
            # Create OBB detection objects without features
            detections = [DetectionOBB(xyxyxyxy_list[i], xywhr_list[i], conf, None)
                          for i, conf in enumerate(confidences) if conf > self.min_confidence]

        # Update tracker
        self.tracker_obb.predict()
        self.tracker_obb.update(detections)

        # Output tracked objects
        outputs = []
        for track in self.tracker_obb.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            xywhr = track.to_xywhr()
            track_id = track.track_id
            # outputs.append(track_id)
            outputs.append(np.array([*xywhr, track_id, track.get_position_history()], dtype=object))

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def _get_features_obb(self, xyxyxyxy_list, ori_img):
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

        for xyxyxyxy in xyxyxyxy_list:
            if self.use_rotated_features:
                # Extract rotated crop aligned to OBB orientation
                try:
                    crop = obb_utils.extract_obb_crop(ori_img, xyxyxyxy, padding=10)
                    im_crops.append(crop)
                except Exception as e:
                    print(f"Warning: Failed to extract rotated crop, using axis-aligned fallback: {e}")
                    # Fallback to axis-aligned crop
                    x_coords = xyxyxyxy[:, 0]
                    y_coords = xyxyxyxy[:, 1]
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
                x_coords = xyxyxyxy[:, 0]
                y_coords = xyxyxyxy[:, 1]
                x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(ori_img.shape[1], x_max)
                y_max = min(ori_img.shape[0], y_max)

                crop = ori_img[y_min:y_max, x_min:x_max]
                im_crops.append(crop)

        if im_crops:
            # features = self.extractor(im_crops)
            # TODO hara: check size
            features = self.extractor.extract_numpy_BGR(im_crops, len(im_crops))
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

#
# class DeepSort(object):
#     def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
#         self.min_confidence = min_confidence # Detection confidence threshold.
#         self.nms_max_overlap = nms_max_overlap # NMS threshold; 1 disables suppression.
#
#         self.extractor = Extractor(model_path, use_cuda=use_cuda) # Extracts features for a batch of image crops.
#
#         max_cosine_distance = max_dist # Maximum cosine distance for cascade matching; larger costs are ignored.
#         nn_budget = 100 # Maximum gallery features per class; older features are removed when this is exceeded.
#         # NearestNeighborDistanceMetric is a nearest-neighbor distance metric.
#         # For each target, it returns the nearest distance to any observed sample
#         # using either Euclidean or cosine distance.
#         # Construct a Tracker from this distance metric.
#         # The first argument can be either 'cosine' or 'euclidean'.
#         metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
#         self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
#
#     def update(self, xywhr, confidences, ori_img):
#         """
#         :param xywhr: shape: (N, 5)
#         :param confidences: shape: (N,)
#         :param ori_img:
#         :return:
#         """
#         self.height, self.width = ori_img.shape[:2]
#         # generate detections
#         # Crop the bbox region from the original image and compute its features.
#         features = self._get_features(bbox_xywh, ori_img)
#         bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
#         # Filter out targets below min_confidence and build the Detection list.
#         detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]
#
#         # run on non-maximum supression
#         # boxes = np.array([d.tlwh for d in detections])
#         # scores = np.array([d.confidence for d in detections])
#         # indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
#         # detections = [detections[i] for i in indices]
#
#         # update tracker
#         self.tracker.predict() # Propagate track state distributions one step forward.
#         self.tracker.update(detections) # Run measurement updates and track management.
#
#         # output bbox identities
#         outputs = []
#         for track in self.tracker.tracks:
#             if not track.is_confirmed() or track.time_since_update > 1:
#                 continue
#             box = track.to_tlwh()
#             x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
#             track_id = track.track_id
#             outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=int))
#         if len(outputs) > 0:
#             outputs = np.stack(outputs,axis=0)
#         return outputs
#
#
#     """
#     TODO:
#         Convert bbox from xc_yc_w_h to xtl_ytl_w_h
#     Thanks JieChen91@github.com for reporting this bug!
#     """
#     # Convert bbox from [x, y, w, h] to [top-left x, top-left y, w, h].
#     @staticmethod
#     def _xywh_to_tlwh(bbox_xywh):
#         if isinstance(bbox_xywh, np.ndarray):
#             bbox_tlwh = bbox_xywh.copy()
#         elif isinstance(bbox_xywh, torch.Tensor):
#             bbox_tlwh = bbox_xywh.clone()
#         bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
#         bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
#         return bbox_tlwh
#
#     # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2].
#     # Some datasets, such as Pascal VOC, use [x, y, w, h] annotations.
#     """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
#     def _xywh_to_xyxy(self, bbox_xywh):
#         x,y,w,h = bbox_xywh
#         x1 = max(int(x-w/2),0)
#         x2 = min(int(x+w/2),self.width-1)
#         y1 = max(int(y-h/2),0)
#         y2 = min(int(y+h/2),self.height-1)
#         return x1,y1,x2,y2
#
#     def _tlwh_to_xyxy(self, bbox_tlwh):
#         """
#         TODO:
#             Convert bbox from xtl_ytl_w_h to xc_yc_w_h
#         Thanks JieChen91@github.com for reporting this bug!
#         """
#         x,y,w,h = bbox_tlwh
#         x1 = max(int(x),0)
#         x2 = min(int(x+w),self.width-1)
#         y1 = max(int(y),0)
#         y2 = min(int(y+h),self.height-1)
#         return x1,y1,x2,y2
#
#     def _xyxy_to_tlwh(self, bbox_xyxy):
#         x1,y1,x2,y2 = bbox_xyxy
#
#         t = x1
#         l = y1
#         w = int(x2-x1)
#         h = int(y2-y1)
#         return t,l,w,h
#
#     # Extract features from cropped image regions.
#     def _get_features(self, bbox_xywh, ori_img):
#         im_crops = []
#         for box in bbox_xywh:
#             x1,y1,x2,y2 = self._xywh_to_xyxy(box)
#             im = ori_img[y1:y2,x1:x2] # Cropped image region.
#             im_crops.append(im)
#         if im_crops:
#             features = self.extractor(im_crops) # Extract features from cropped image regions.
#         else:
#             features = np.array([])
#         return features
