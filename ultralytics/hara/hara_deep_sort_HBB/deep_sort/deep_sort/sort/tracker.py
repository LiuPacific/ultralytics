# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from scipy.optimize import linear_sum_assignment
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track, TrackState
from ultralytics.hara.hara_deep_sort_HBB.deep_sort.configs.common_cfg import cfg
import os

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3, csv_path=None, flush_interval=30,
                 tracking_csv_path="tracking_output.csv"):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.MAX_ID_POOL = cfg.DEEPSORT.get("MAX_ID_POOL", 0)
        self.reuse_id_assignment_distance_threshold = cfg.DEEPSORT.get(
            "REUSE_ID_ASSIGNMENT_DISTANCE_THRESHOLD", 200
        )

        if kalman_filter is not None:
            self.kf = kalman_filter.KalmanFilter()
        else:
            from kalman_filter import  KalmanFilter
            self.kf = KalmanFilter()

        self.tracks = []   # List of tracks.
        self._next_id = 1  # Next track ID to assign.
        # CSV logging buffer and configuration
        self.tracking_csv_path = tracking_csv_path
        if self.tracking_csv_path and os.path.exists(self.tracking_csv_path):
            os.remove(self.tracking_csv_path)
        self._csv_buffer = []
        self._frames_since_flush = 0
        self._flush_interval = flush_interval
        self.frame_id = 0  # Global frame counter

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Increment global frame counter
        self.frame_id += 1

        if self.frame_id >= 200 and self.frame_id <= 204:
            print("---")

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        
        # 1. Handle matched results.
        for track_idx, detection_idx in matches:
            # Update the corresponding detection in tracks.
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        
        # 2. Mark unmatched tracks as missed.
        # If a track misses while Tentative, delete it; also delete stale tracks.
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # 3. Initialize tracks for unmatched detections.
        for detection_idx in unmatched_detections:
            if cfg.DEEPSORT.get("USE_OPTIMIZATION", False) and cfg.DEEPSORT.get("REUSE_ID", False):
                self._initiate_track_MAX_ID_POOL(
                    detections[detection_idx],
                    MAX_ID_POOL=self.MAX_ID_POOL,
                )
            else:
                self._initiate_track(detections[detection_idx])

        # If MAX_ID_POOL is 0, use original behavior (delete tracks after max_age)
        # if not cfg.DEEPSORT.USE_OPTIMIZATION or self.MAX_ID_POOL == 0 or not cfg.DEEPSORT.get("REUSE_ID", False):
        if not cfg.DEEPSORT.USE_OPTIMIZATION:
            self.tracks = [t for t in self.tracks if not t.is_deleted()]

        if cfg.DEEPSORT.get("USE_OPTIMIZATION", False):
            if cfg.DEEPSORT.get("TRACK_RECONNECTION_ON", True):
                # Try to reconnect a fresh replacement track to an older lost one.
                # This is a lightweight spatial re-identification step and runs
                # after normal association has already finished.
                if self.MAX_ID_POOL > 0:
                    self._reidentify_tracks_MAX_ID_POOL()
                else:
                    self._reidentify_tracks()

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            # Get all confirmed track IDs.
            if not track.is_confirmed():
                continue
            features += track.features # Add features from confirmed tracks to the features list.
            # Get the track_id corresponding to each feature.
            targets += [track.track_id for _ in track.features]
            track.features = []

        # Update the feature set in the distance metric.
        if cfg.DEEPSORT.get("USE_REID", False):
            self.metric.partial_fit(
                np.asarray(features), np.asarray(targets), active_targets)

        # save tracking information
        self._save_tracking_information()

    def _save_tracking_information(self):
        """Record tracking information for current frame into internal buffer and flush to CSV when needed."""
        self.record_frame(self.frame_id)
        self._frames_since_flush += 1
        if self._frames_since_flush >= self._flush_interval:
            # flush
            self._frames_since_flush = 0
            self.save_csv()

    def record_frame(self, global_frame_id: int):
        """Record current tracks into internal CSV buffer for the given global frame id.
        Only records detected tracks (time_since_update == 0) for non-optimized mode.

        This function will flush the buffer to disk every `self._flush_interval` frames.
        """
        for track in self.tracks:
            # Only record detected tracks (confirmed + detected in current frame)
            if track.time_since_update != 0:
                continue

            tid = int(track.track_id)
            detected = True  # By definition, we only record detected tracks

            history = track.get_position_history()
            if not history:
                continue
            try:
                # xyah format: x, y, aspect_ratio, height
                x, y, a, h = (
                    float(history[-1][0]),
                    float(history[-1][1]),
                    float(history[-1][2]),
                    float(history[-1][3]),
                )
                w = a * h  # width = aspect_ratio * height
                confidence = float(track.last_confidence) if track.last_confidence is not None else None
            except (TypeError, IndexError):
                continue

            # For HBB (horizontal bounding box), no rotation angle
            # CSV format: frame_id, track_id, center_x, center_y, width, height, confidence, detected
            self._csv_buffer.append((int(global_frame_id), tid, x, y, w, h, confidence, bool(detected)))



    def save_csv(self, file_path: str = None, force: bool = False):
        """Flush internal CSV buffer to disk.

        If file_path is None, use self.tracking_csv_path.
        """
        import os
        import csv

        if not self._csv_buffer and not force:
            return

        out_path = file_path or self.tracking_csv_path
        file_exists = os.path.exists(out_path)

        # write rows
        with open(out_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['frame_id', 'track_id', 'center_x', 'center_y', 'width', 'height', 'confidence', 'detected'])
            for row in self._csv_buffer:
                # Convert None to empty string for CSV
                writer.writerow([r if r is not None else '' for r in row])

        # clear buffer
        self._csv_buffer = []

    def _record_track_detection(self, track, measurement, detection):
        track.position_history.append(np.asarray(measurement, dtype=float).copy())
        try:
            track.last_confidence = detection.confidence
        except Exception:
            track.last_confidence = None

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])

            cost_matrix = np.ones((len(targets), len(features)))
            if features[0] is not None:
                cost_matrix = self.metric.distance(features, targets)

            # Compute the gated cost matrix.
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        # Separate confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        # Run cascade matching on confirmed tracks and return matched tracks,
        # unmatched tracks, and unmatched detections.
        # matching_cascade matches detection boxes to confirmed tracks by feature
        # using the gated cost matrix.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.        
        # Combine unconfirmed tracks and tracks that just failed to match into
        # iou_track_candidates, then run IoU-based matching.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1] # Tracks that just failed to match.
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1] # Tracks that did not just fail to match.
        # Run IoU matching for targets that did not match in cascade matching.
        # min_cost_matching solves the linear assignment problem with the
        # Hungarian algorithm. It uses iou_cost to associate remaining and
        # unconfirmed tracks.
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b # Combine the two matching stages.
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        measurement = detection.to_xyah()
        mean, covariance = self.kf.initiate(measurement)
        new_track = Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature)
        self._record_track_detection(new_track, measurement, detection)
        self.tracks.append(new_track)
        # hara change starts;
        if self._next_id >=16:
            # if next id is 16, it means currently we have 15 ids.
            print("16")
            # if the detection couldn't match any existing track, it means the track feature's similarity couldn't match the detection's feature.

        # hara change ends;
        self._next_id += 1

    def _initiate_track_MAX_ID_POOL(self, detection, MAX_ID_POOL=15):
        """Create a new HBB track while reusing IDs from a fixed pool.

        Behavior:
        - Prefer reusing the ID of a missed track whose last center is closest
          to the new detection, if it is within
          `REUSE_ID_ASSIGNMENT_DISTANCE_THRESHOLD`.
        - Otherwise, assign the smallest unused ID from `1..MAX_ID_POOL`.
        - If the pool is saturated and nothing is close enough to reuse, skip
          creating a new track. That keeps the active identity space bounded.
        """
        measurement = detection.to_xyah()
        mean, covariance = self.kf.initiate(measurement)

        used_ids = {
            track.track_id
            for track in self.tracks
            if 1 <= track.track_id <= MAX_ID_POOL
        }

        detection_center = np.asarray(measurement[:2], dtype=float)
        reuse_id = None
        best_candidate = None
        best_distance = np.inf

        # First try to revive a recently missed track from the pool. This is
        # the safest reuse case because the old identity is still spatially
        # close and has not fully aged out.
        missed_candidates = [
            track for track in self.tracks
            if 1 <= track.track_id <= MAX_ID_POOL
            and not track.is_deleted()
            and track.time_since_update > 0
        ]

        for track in missed_candidates:
            history = track.get_position_history()
            if history:
                last_center = np.asarray(history[-1][:2], dtype=float)
            else:
                last_center = np.asarray(track.mean[:2], dtype=float)

            distance = np.linalg.norm(last_center - detection_center)
            if distance < best_distance:
                best_distance = distance
                best_candidate = track

        if (
            best_candidate is not None
            and best_distance < self.reuse_id_assignment_distance_threshold
        ):
            reuse_id = best_candidate.track_id

        # If no nearby missed track exists, take the smallest unused ID in the pool.
        if reuse_id is None:
            for candidate_id in range(1, MAX_ID_POOL + 1):
                if candidate_id not in used_ids:
                    reuse_id = candidate_id
                    break

        # When every pool ID is already occupied and none is close enough to
        # reuse, do not create another identity outside the configured pool.
        if reuse_id is None:
            return

        new_track = Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature)

        old_track = None
        for track in self.tracks:
            if track.track_id == reuse_id:
                old_track = track
                break

        if old_track is not None:
            old_track.merge_with(new_track)
            new_track.track_id = old_track.track_id
            old_track.state = TrackState.Deleted
            self.tracks.remove(old_track)
        else:
            new_track.track_id = reuse_id

        self._record_track_detection(new_track, measurement, detection)

        self.tracks.append(new_track)
        self._next_id += 1

    def _reidentify_tracks(self):
        """Reconnect newly created HBB tracks to older lost tracks by center distance.

        Rationale:
        - A missed association can create a new track for the same chicken.
        - When that new track is still very young, we can compare its first
          observed center against the last center of older unmatched tracks.
        - The Hungarian assignment gives a globally consistent one-to-one
          pairing, and only pairs below the configured distance threshold
          are accepted.
        """
        new_tracks = [t for t in self.tracks if (t.age>=3 and t.age <=5)]
        if not new_tracks:
            return

        # Only consider tracks that have been unmatched for more than one frame,
        # otherwise the normal matcher should still be handling them.
        unmatched_tracks = [
            t for t in self.tracks
            if t.time_since_update > 1 and t not in new_tracks
        ]
        if not unmatched_tracks:
            return

        reconnection_distance_threshold = cfg.DEEPSORT.get("RECONNECTION_DISTANCE_THRESHOLD", 200)
        cost_matrix = np.full((len(unmatched_tracks), len(new_tracks)), np.inf, dtype=float)

        for i, unmatched_track in enumerate(unmatched_tracks):
            unmatched_center = self._track_last_detection_center(unmatched_track)

            for j, new_track in enumerate(new_tracks):
                new_center = self._track_first_detection_center(new_track)
                cost_matrix[i, j] = np.linalg.norm(unmatched_center - new_center)

        # Use Hungarian algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Reassign IDs for matches below distance threshold
        for row_index, col_index in zip(row_ind, col_ind):
            if cost_matrix[row_index, col_index] >= reconnection_distance_threshold:
                continue

            old_track = unmatched_tracks[row_index]
            new_track = new_tracks[col_index]

            # Preserve the old identity and history, but keep the new track's
            # current motion state because it reflects the latest detection.
            old_track.merge_with(new_track)
            new_track.track_id = old_track.track_id
            old_track.state = TrackState.Deleted

            if old_track in self.tracks:
                self.tracks.remove(old_track)

    def _reidentify_tracks_MAX_ID_POOL(self):
        """Reconnect overflow new tracks back to older tracks inside the ID pool.

        This keeps the optimization behavior of `_reidentify_tracks`, but
        applies the fixed-ID-pool constraint:
        - new replacement tracks must have `track_id > MAX_ID_POOL`;
        - old candidate tracks must have `track_id` in `1..MAX_ID_POOL`;
        - accepted matches inherit the old in-pool identity.
        """
        max_id_pool = int(self.MAX_ID_POOL)
        if max_id_pool <= 0:
            print("max id pool is {}".format(max_id_pool))
            return

        new_tracks = [
            track for track in self.tracks
            if track.track_id > max_id_pool
            and 3 <= track.age <= 100
            and not track.is_deleted()
        ]
        if not new_tracks:
            return

        unmatched_tracks = [
            track for track in self.tracks
            if 1 <= track.track_id <= max_id_pool
            and track.time_since_update > 1
            and track not in new_tracks
        ]
        if not unmatched_tracks:
            return

        reconnection_distance_threshold = cfg.DEEPSORT.get("RECONNECTION_DISTANCE_THRESHOLD", 200)
        cost_matrix = np.full((len(unmatched_tracks), len(new_tracks)), np.inf, dtype=float)

        for i, unmatched_track in enumerate(unmatched_tracks):
            unmatched_center = self._track_last_detection_center(unmatched_track)
            for j, new_track in enumerate(new_tracks):
                new_center = self._track_first_detection_center(new_track)
                cost_matrix[i, j] = np.linalg.norm(unmatched_center - new_center)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for row_index, col_index in zip(row_ind, col_ind):
            if cost_matrix[row_index, col_index] >= reconnection_distance_threshold:
                continue

            old_track = unmatched_tracks[row_index]
            new_track = new_tracks[col_index]

            old_track.merge_with(new_track)
            new_track.track_id = old_track.track_id
            old_track.state = TrackState.Deleted

            if old_track in self.tracks:
                self.tracks.remove(old_track)

    def _track_last_detection_center(self, track):
        history = track.get_position_history()
        if history:
            return np.asarray(history[-1][:2], dtype=float)
        return np.asarray(track.mean[:2], dtype=float)

    def _track_first_detection_center(self, track):
        history = track.get_position_history()
        if history:
            return np.asarray(history[0][:2], dtype=float)
        return np.asarray(track.mean[:2], dtype=float)
