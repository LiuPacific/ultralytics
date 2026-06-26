from __future__ import absolute_import
import os
import numpy as np

from .iou_matching_obb import obb_iou_cost, iou_cost_fallback
from .track_obb import TrackOBB, TrackState
from .linear_assignment_obb import gate_cost_matrix_obb, matching_cascade, min_cost_matching
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_distances
from ultralytics.hara.hara_obb_deepsort4.deep_sort.configs.common_cfg import cfg


class TrackerOBB:
    """
    Extended tracker for OBB (Oriented Bounding Box) detections.
    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3, kalman_filter=None, use_rotated_iou=True,
                 MAX_ID_POOL=0, reconnection_distance_threshold=400, reuse_id_assignment_distance_threshold=200,
                 flush_interval=30, tracking_csv_path="tracking_output.csv"):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.use_rotated_iou = use_rotated_iou
        self.MAX_ID_POOL = MAX_ID_POOL
        self.reconnection_distance_threshold = reconnection_distance_threshold
        self.reuse_id_assignment_distance_threshold = reuse_id_assignment_distance_threshold
        if kalman_filter is not None:
            self.kf = kalman_filter
        else:
            from kalman_filter_obb import KalmanFilterOBB
            self.kf = KalmanFilterOBB()

        self.tracks = []
        self._next_id = 1
        self.frame_id = 0  # Global frame counter (incremented each update)
        # CSV logging buffer and configuration
        self.tracking_csv_path = tracking_csv_path
        if self.tracking_csv_path and os.path.exists(self.tracking_csv_path):
            os.remove(self.tracking_csv_path)
        self._csv_buffer = []
        self._frames_since_flush = 0
        self._flush_interval = flush_interval

    def predict(self):
        """Propagate track state distributions one time step forward."""
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management for OBB detections."""
        # Increment global frame counter
        self.frame_id += 1

        # 201 gating distance 0.6
        # 203 5.79
        # 204
        if self.frame_id >= 200 and self.frame_id <= 204:
            print("---")

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx], self.frame_id)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            if cfg.DEEPSORT.USE_OPTIMIZATION and self.MAX_ID_POOL > 0:
                self._initiate_track_MAX_ID_POOL(detections[detection_idx], MAX_ID_POOL=self.MAX_ID_POOL)
            else:
                self._initiate_track(detections[detection_idx])

        # If MAX_ID_POOL is 0, use original behavior (delete tracks after max_age)
        if cfg.DEEPSORT.USE_OPTIMIZATION and self.MAX_ID_POOL == 0:
            self.tracks = [t for t in self.tracks if not t.is_deleted()]

        if cfg.DEEPSORT.USE_OPTIMIZATION:
            # Perform track re-identification
            self._reidentify_tracks()
            # Perform track re-identification by ReID (appearance features)
            self._reidentify_tracks_by_ReID()

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

        # save tracking information
        self._save_tracking_information()

    def _save_tracking_information(self):
        """Record tracking information for current frame into internal buffer and flush to CSV when needed."""
        self.record_frame_into_buffer(self.frame_id)
        self._frames_since_flush += 1
        if self._frames_since_flush >= self._flush_interval:
            # flush
            self._frames_since_flush = 0
            self.save_csv()

    def record_frame_into_buffer(self, global_frame_id: int):
        """Record current tracks into internal CSV buffer for the given global frame id.

        This function will flush the buffer to disk every `self._flush_interval` frames.
        """
        for track in self.tracks:
            tid = int(track.track_id)
            # detected this frame if time_since_update == 0
            detected = (track.time_since_update == 0)

            # If optimization is not used, don't record the track's data. There should be new tracks created and recorded.
            # if not cfg.DEEPSORT.USE_OPTIMIZATION or self.MAX_ID_POOL == 0:
            #     continue

            # Determine center coordinates: use last detected xy if available, otherwise None
            if track.last_detected_xywhr is not None:
                try:
                    cx, cy = float(track.last_detected_xywhr[0]), float(track.last_detected_xywhr[1])
                except (TypeError, IndexError):
                    cx, cy = (None, None)
            else:
                cx, cy = (None, None)

            # if cfg.DEEPSORT.USE_OPTIMIZATION and self.MAX_ID_POOL>0:
            if detected and track.last_detected_xywhr is not None:
                try:
                    w, h, angle = (float(track.last_detected_xywhr[2]), float(track.last_detected_xywhr[3]),
                                   float(track.last_detected_xywhr[4]))
                    confidence = float(track.last_confidence) if track.last_confidence is not None else None
                except (TypeError, IndexError):
                    w, h, angle, confidence = (None, None, None, None)
            else:
                # not detected in this frame: keep center from last detected, other fields null
                w, h, angle, confidence = (None, None, None, None)

            self._csv_buffer.append((int(global_frame_id), tid, cx, cy, w, h, angle, confidence, bool(detected)))

    def save_csv(self, file_path: str = None, force: bool = False):
        """Flush internal CSV buffer to disk.

        If file_path is None, use self.csv_path.
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
                writer.writerow(
                    ['frame_id', 'track_id', 'center_x', 'center_y', 'width', 'height', 'angle', 'confidence',
                     'detected'])
            for row in self._csv_buffer:
                # Convert None to empty string for CSV
                writer.writerow([r if r is not None else '' for r in row])

        # clear buffer
        self._csv_buffer = []

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])

            cost_matrix = np.ones((len(targets), len(features)))
            if features[0] is not None:
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
            matching_cascade(
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
            min_cost_matching(
                iou_cost_func, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        # For OBB detections, we need to convert to xywhr format for Kalman filter
        xywhr = detection.to_xywhr()
        mean, covariance = self.kf.initiate(xywhr)
        new_track = TrackOBB(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, self.frame_id)
        # set last detected info on the newly created track
        new_track.last_detected_xywhr = xywhr
        new_track.last_confidence = detection.confidence
        self.tracks.append(new_track)
        self._next_id += 1

    def _initiate_track_MAX_ID_POOL(self, detection, MAX_ID_POOL=15):
        """
        Initiate a new track for a detection, but reuse track IDs from a fixed pool
        of IDs (1..MAX_ID_POOL) when possible. If there are already MAX_ID_POOL
        active (not deleted) tracks with IDs in the pool, do NOT create a new
        track (this avoids creating > MAX_ID_POOL simultaneous tracks). When
        possible, prefer to reuse an ID from a missed track (time_since_update>0).
        """

        # For OBB detections, convert to xywhr format for Kalman filter
        xywhr = detection.to_xywhr()
        mean, covariance = self.kf.initiate(xywhr)

        if self.frame_id == 40:
            print("---")

        # Determine currently used IDs within the pool (active & not deleted)
        used_ids = set([t.track_id for t in self.tracks if not t.is_deleted() and 1 <= t.track_id <= MAX_ID_POOL])

        # # Count active (not deleted) tracks within pool
        # active_count = len(used_ids)
        #
        # # If pool is full, do not create a new track
        # if active_count >= MAX_ID_POOL:
        #     # No available ID in the pool; skip creating a new track
        #     # This prevents creating spurious duplicate tracks when many detections
        #     # fall on the same object and the pool is saturated.
        #     # Optionally, could attempt to match/merge here; keep simple and skip.
        #     # print("Track pool full: skipping initiation of a new track")
        #     return

        # Try to find a deleted/missed track to reuse its ID based on spatial proximity
        # Prefer deleted tracks (those that exceeded max_age and were marked deleted),
        # matching by the last known position. If none within threshold, try missed
        # (time_since_update>0) tracks by distance. Do NOT remove deleted tracks from
        # the list — we keep them as history for future matching.
        reuse_id = None

        # detection center for distance comparisons
        try:
            detection_center = np.array(xywhr[:2], dtype=float)
        except Exception as e:
            print("exception ", e)
            return
            # detection_center = np.array([0.0, 0.0], dtype=float)

        # both deleted and missed are considered equally.
        missed_candidates = [t for t in self.tracks if 1 <= t.track_id <= MAX_ID_POOL and t.time_since_update > 0]
        best_candidate = None
        best_dist = np.inf
        for t in missed_candidates:
            hist = t.get_position_history()
            if hist:
                last_center = np.array(hist[-1][:2], dtype=float)
            else:
                try:
                    last_center = np.array(t.to_xywhr()[:2], dtype=float)
                except Exception as e:
                    print("exception ", e)
                    continue
            dist = np.linalg.norm(last_center - detection_center)
            if dist < best_dist:
                best_dist = dist
                best_candidate = t

        if best_candidate is not None and best_dist < self.reuse_id_assignment_distance_threshold:
            reuse_id = best_candidate.track_id

        # # 1) Consider deleted tracks first
        # deleted_candidates = [t for t in self.tracks if t.is_deleted() and 1 <= t.track_id <= MAX_ID_POOL]
        # best_candidate = None
        # best_dist = np.inf
        # for t in deleted_candidates:
        #     hist = t.get_position_history()
        #     if hist:
        #         last_center = np.array(hist[-1][:2], dtype=float)
        #     else:
        #         # fallback to track's current state if history not available
        #         try:
        #             last_center = np.array(t.to_xywhr()[:2], dtype=float)
        #         except Exception as e:
        #             print("exception ", e)
        #             continue
        #     dist = np.linalg.norm(last_center - detection_center)
        #     if dist < best_dist:
        #         best_dist = dist
        #         best_candidate = t
        #
        # if best_candidate is not None and best_dist < self.reuse_id_assignment_distance_threshold:  #
        #     reuse_id = best_candidate.track_id
        #
        # # 2) Fallback: consider missed (un-deleted) candidates and pick nearest by distance
        # if reuse_id is None:
        #     missed_candidates = [t for t in self.tracks if
        #                          (not t.is_deleted()) and 1 <= t.track_id <= MAX_ID_POOL and t.time_since_update > 0]
        #     best_candidate = None
        #     best_dist = np.inf
        #     for t in missed_candidates:
        #         hist = t.get_position_history()
        #         if hist:
        #             last_center = np.array(hist[-1][:2], dtype=float)
        #         else:
        #             try:
        #                 last_center = np.array(t.to_xywhr()[:2], dtype=float)
        #             except Exception as e:
        #                 print("exception ", e)
        #                 continue
        #         dist = np.linalg.norm(last_center - detection_center)
        #         if dist < best_dist:
        #             best_dist = dist
        #             best_candidate = t
        #
        #     if best_candidate is not None and best_dist < self.reuse_id_assignment_distance_threshold:
        #         reuse_id = best_candidate.track_id

        # if reuse_id is None:
        #     return
        # self.tracks.append(
        #     TrackOBB(mean, covariance, reuse_id, self.n_init, self.max_age, detection.feature, self.frame_id)
        # )

        # If we didn't reuse an ID, pick the smallest unused ID in pool
        if reuse_id is None:
            for i in range(1, MAX_ID_POOL + 1):
                if i not in used_ids:
                    reuse_id = i
                    break

        # Fallback: if reuse_id still None (shouldn't happen), use next id
        # if reuse_id is None:
        #     reuse_id = self._next_id
        if reuse_id is None:
            return

        # Create new track with temporary ID first
        temp_id = self._next_id
        new_track = TrackOBB(
            mean, covariance, temp_id, self.n_init, self.max_age,
            detection.feature, self.frame_id)

        # Find the old track with reuse_id to merge into new_track
        old_track = None
        if reuse_id is not None:
            for t in self.tracks:
                if t.track_id == reuse_id and t != new_track:
                    old_track = t
                    break

        # If we found an old track with the reuse_id, merge it into new_track
        if old_track is not None:
            print(f"Reusing track ID {reuse_id}: merging old track into new detection")
            # Merge old track history into new track
            old_track.merge_with(new_track)
            # Transfer the old track ID to new track
            new_track.track_id = old_track.track_id
            # Mark the old track as deleted
            old_track.state = TrackState.Deleted
            self.tracks.remove(old_track)
        else:
            # No old track to merge; just use the reuse_id directly
            new_track.track_id = reuse_id

        # set last detected info on the newly created track
        new_track.last_detected_xywhr = xywhr
        new_track.last_confidence = detection.confidence
        self.tracks.append(new_track)
        self._next_id += 1

    def _reidentify_tracks(self):
        """Re-identify tracks that disappeared and reappeared using position history."""
        # Find new tracks that have existed for exactly 3 frames
        # duplicate detections exist more than 3 frames
        new_tracks = [t for t in self.tracks if t.age == 3]

        if not new_tracks:
            return

        # Find unmatched tracks that disappeared more than 1 frame ago and are not new tracks
        unmatched_tracks = [t for t in self.tracks if t.time_since_update > 1 and t not in new_tracks]

        if not unmatched_tracks:
            return

        # Create cost matrix: distance between last position of unmatched and first position of new
        num_unmatched = len(unmatched_tracks)
        num_new = len(new_tracks)
        cost_matrix = np.full((num_unmatched, num_new), np.inf)

        for i, unmatched_track in enumerate(unmatched_tracks):
            # Last position of unmatched track
            history = unmatched_track.get_position_history()
            if history:
                unmatched_center = history[-1][:2]  # cx, cy of last position
            else:
                unmatched_center = unmatched_track.to_xywhr()[:2]  # fallback
            for j, new_track in enumerate(new_tracks):
                # First position of new track
                history = new_track.get_position_history()
                if history:
                    new_center = history[0][:2]  # cx, cy of first position
                else:
                    new_center = new_track.to_xywhr()[:2]  # fallback
                distance = np.linalg.norm(unmatched_center - new_center)
                cost_matrix[i, j] = distance

        # Use Hungarian algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Reassign IDs for matches below distance threshold
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.reconnection_distance_threshold:
                old_track = unmatched_tracks[r]
                new_track = new_tracks[c]
                print(f"Re-identifying track {new_track.track_id} as {old_track.track_id}")

                # Merge old track into new track, then delete the old one
                old_track.merge_with(new_track)
                new_track.track_id = old_track.track_id
                old_track.state = TrackState.Deleted
                self.tracks.remove(old_track)

    def _reidentify_tracks_by_ReID(self):
        """
        Re-identify lost tracks by comparing appearance features with new tracks.
        
        Process:
        1. Find unmatched tracks that have been lost for 6 seconds (180 frames at 30 FPS)
        2. Find new tracks that have been alive for 5 seconds (150 frames at 30 FPS)
        3. Compare appearance features using cosine similarity
        4. Use Hungarian algorithm to assign new tracks to lost tracks based on feature similarity
        
        Parameters
        ----------
        FPS : int
            Frames per second (used to convert time thresholds)
        """
        FPS = 30  # Default FPS for chicken tracking

        # Time thresholds
        lost_track_threshold = int(4 * FPS)  # 120 frames at 30 FPS
        track_top_age_to_be_fresh = int(3 * FPS)  # 90 frames at 30 FPS

        # Find lost tracks (unmatched for 6 seconds)
        lost_tracks = [t for t in self.tracks
                       if t.time_since_update >= lost_track_threshold
                       and t.is_confirmed()]

        if not lost_tracks:
            return

        # Find new tracks (alive for at least 5 seconds)
        new_tracks = [t for t in self.tracks
                      if t.age <= track_top_age_to_be_fresh
                      and t.is_confirmed()
                      and t.time_since_update == 0]

        if not new_tracks:
            return

        num_lost = len(lost_tracks)
        num_new = len(new_tracks)

        # Build cost matrix using appearance feature similarity (cosine distance)
        cost_matrix = np.full((num_lost, num_new), np.inf)

        for i, lost_track in enumerate(lost_tracks):
            lost_features = lost_track.get_appearance_features()
            # Always get position history (may be empty) for spatial fallback
            lost_history = lost_track.get_position_history()

            for j, new_track in enumerate(new_tracks):
                new_features = new_track.get_appearance_features()
                # Always get position history (may be empty) for spatial fallback
                new_history = new_track.get_position_history()

                # Calculate feature-based cost (cosine distance) using sklearn
                if lost_features and new_features:
                    try:
                        # Build 2D arrays of flattened features
                        lost_arr = np.vstack([f['feature'].flatten() for f in lost_features])
                        new_arr = np.vstack([f['feature'].flatten() for f in new_features])

                        # Compute pairwise cosine distances and take the minimum
                        dists = cosine_distances(lost_arr.astype(float), new_arr.astype(float))
                        cost_matrix[i, j] = float(np.min(dists))
                    except Exception:
                        # Fallback to previous pairwise method if something goes wrong
                        similarity_scores = []
                        for lost_feat_data in lost_features:
                            lost_feat = lost_feat_data['feature']
                            for new_feat_data in new_features:
                                new_feat = new_feat_data['feature']
                                lost_feat_flat = lost_feat.flatten() if hasattr(lost_feat, 'flatten') else lost_feat
                                new_feat_flat = new_feat.flatten() if hasattr(new_feat, 'flatten') else new_feat
                                lost_feat_norm = lost_feat_flat / (np.linalg.norm(lost_feat_flat) + 1e-8)
                                new_feat_norm = new_feat_flat / (np.linalg.norm(new_feat_flat) + 1e-8)
                                similarity = np.dot(lost_feat_norm, new_feat_norm)
                                distance = 1.0 - similarity
                                similarity_scores.append(distance)
                        if similarity_scores:
                            cost_matrix[i, j] = min(similarity_scores)
                else:
                    print("---------features not available-----")
                    return
                    # # Fallback to spatial distance if features not available
                    # if lost_history and new_history:
                    #     lost_center = np.array(lost_history[-1][:2])
                    #     new_center = np.array(new_history[0][:2])
                    #     spatial_distance = np.linalg.norm(lost_center - new_center)
                    #     # Normalize spatial distance (scale to [0, 1])
                    #     cost_matrix[i, j] = spatial_distance / 1000.0  # 1000 pixel threshold

        # Use Hungarian algorithm to find optimal assignment

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Reassign IDs for matches below similarity threshold
        # Cosine distance threshold: 0.3 (corresponds to ~60 degree angle or 0.3 similarity)
        similarity_threshold = 0.3
        matched_new_track_indices = set()

        for r, c in zip(row_ind, col_ind):
            # if cost_matrix[r, c] < similarity_threshold and cost_matrix[r, c] != np.inf:
            if cost_matrix[r, c] < similarity_threshold and cost_matrix[r, c] != np.inf and cost_matrix[
                r, c] < self.reconnection_distance_threshold:
                lost_track = lost_tracks[r]
                new_track = new_tracks[c]

                # Merge the lost track with the new track
                print(f"ReID: Reconnecting track {new_track.track_id} with lost track {lost_track.track_id}")

                # Merge old track history (positions & features) into new track
                lost_track.merge_with(new_track)

                # Transfer the old track ID to the new track
                new_track.track_id = lost_track.track_id

                # Mark the lost track as deleted
                lost_track.state = TrackState.Deleted

                matched_new_track_indices.add(c)
