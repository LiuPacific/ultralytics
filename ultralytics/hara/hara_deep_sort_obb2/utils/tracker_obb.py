# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
import sys
import os
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))
from ultralytics.hara.hara_obb_deepsort.deep_sort.deep_sort.sort import linear_assignment
from .iou_matching_obb import obb_iou_cost, iou_cost_fallback
from .track_obb import TrackOBB, TrackState
from .linear_assignment_obb import gate_cost_matrix_obb


class TrackerOBB:
    """
    Extended tracker for OBB (Oriented Bounding Box) detections.
    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3, kalman_filter=None, use_rotated_iou=True):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.use_rotated_iou = use_rotated_iou

        if kalman_filter is not None:
            self.kf = kalman_filter
        else:
            from kalman_filter_obb import KalmanFilterOBB
            self.kf = KalmanFilterOBB()

        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward."""
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management for OBB detections."""
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])

            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = gate_cost_matrix_obb(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]

        # Choose appropriate IOU cost function
        if self.use_rotated_iou:
            iou_cost_func = obb_iou_cost
        else:
            iou_cost_func = iou_cost_fallback

        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_cost_func, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        # For OBB detections, we need to convert to xywhr format for Kalman filter
        xywhr = detection.to_xywhr()
        mean, covariance = self.kf.initiate(xywhr)
        self.tracks.append(TrackOBB(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
