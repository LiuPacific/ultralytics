from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics.trackers.utils import matching
from ultralytics.hara.hara_byte_track.config.common_cfg import cfg

class HaraBYTETracker(BYTETracker):
    """Hara-owned ByteTrack tracker.

    This class is registered as a separate Ultralytics tracker type so Hara
    experiments can customize ByteTrack behavior without editing the upstream
    `BYTETracker` implementation.
    """

    DETECTION_COUNT_SAMPLE_INTERVAL = 15
    DETECTION_COUNT_STABILITY_WINDOW = 3

    def __init__(self, args: Any):
        super().__init__(args)
        self.runtime_cfg = cfg.get("ByteTrack", {})
        self.bbox_history = []
        self.frame_memory_length = self.runtime_cfg.get("FRAME_MEMORY_LENGTH", 5)
        self.detection_frame_count = 0
        self.detection_count_history = []
        self.detection_optimization_target = None
        self.position_history_length = self.runtime_cfg.get("POSITION_HISTORY_LENGTH", 300)
        self.position_history = {}
        self.detection_region = (
            self.runtime_cfg.get("DETECTION_REGION_X_MIN", 270),
            self.runtime_cfg.get("DETECTION_REGION_X_MAX", 1900),
            self.runtime_cfg.get("DETECTION_REGION_Y_MIN", 100),
            self.runtime_cfg.get("DETECTION_REGION_Y_MAX", 1600),
        )

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
        """Update tracker state.

        This local copy intentionally starts from Ultralytics `BYTETracker`
        behavior. Customize the association, lifecycle, or ID-assignment steps
        here for Hara-specific ByteTrack experiments.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        results, feats = self._filter_detections_by_region(results, feats)

        # hara detection optimization
        if self.runtime_cfg.get("USE_OPTIMIZATION", False) and self.runtime_cfg.get("DETECTION_OPTIMIZATION_ON", False):
            results, feats = self._apply_detection_optimization(results, img, feats)

        scores = results.conf
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        results_second = results[inds_second]
        results = results[remain_inds]
        feats_keep = feats_second = img
        if feats is not None and len(feats):
            feats_keep = feats[remain_inds]
            feats_second = feats[inds_second]

        detections = self.init_track(results, feats_keep)
        unconfirmed = []
        tracked_stracks: list[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)
        if hasattr(self, "gmc") and img is not None:
            try:
                warp = self.gmc.apply(img, results.xyxy)
            except Exception:
                warp = np.eye(2, 3)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                self._record_detection(track, det)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                self._record_detection(track, det)
                refind_stracks.append(track)

        detections_second = self.init_track(results_second, feats_second)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections_second)
        matches, u_track, _u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                self._record_detection(track, det)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                self._record_detection(track, det)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            self._record_detection(unconfirmed[itracked], detections[idet])
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            self._record_detection(track, track)
            activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_frames_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]

        if self.runtime_cfg.get("USE_OPTIMIZATION", False):
            if self.runtime_cfg.get("TRACK_RECONNECTION_ON", False):
                if self._max_id_pool() > 0:
                    self._reidentify_tracks_MAX_ID_POOL()
                else:
                    self._reidentify_tracks()

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def _reidentify_tracks(self):
        """Reconnect fresh replacement tracks to older lost/removed tracks by center distance."""
        new_tracks = [
            track for track in self.tracked_stracks
            if 3 <= self._track_age(track) <= 5
        ]
        if not new_tracks:
            return

        active_ids = {track.track_id for track in self.tracked_stracks}
        unmatched_tracks = self._reconnection_candidates(active_ids)
        if not unmatched_tracks:
            return

        reconnection_distance_threshold = self.runtime_cfg.get("RECONNECTION_DISTANCE_THRESHOLD", 200)
        cost_matrix = np.full((len(unmatched_tracks), len(new_tracks)), np.inf, dtype=float)

        for i, unmatched_track in enumerate(unmatched_tracks):
            unmatched_center = self._track_last_detection_center(unmatched_track)
            for j, new_track in enumerate(new_tracks):
                new_center = self._track_first_detection_center(new_track)
                cost_matrix[i, j] = np.linalg.norm(unmatched_center - new_center)

        matches, _, _ = matching.linear_assignment(cost_matrix, thresh=reconnection_distance_threshold)
        for row_index, col_index in matches:
            old_track = unmatched_tracks[int(row_index)]
            new_track = new_tracks[int(col_index)]
            old_track_id = old_track.track_id
            new_track_id = new_track.track_id

            new_track.track_id = old_track_id
            self._merge_position_history(old_track_id, new_track_id)
            new_track.start_frame = old_track.start_frame
            new_track.tracklet_len += getattr(old_track, "tracklet_len", 0)
            self.lost_stracks = [track for track in self.lost_stracks if track.track_id != old_track_id]
            self.removed_stracks = [track for track in self.removed_stracks if track.track_id != old_track_id]
            old_track.mark_removed()

    def _reidentify_tracks_MAX_ID_POOL(self):
        """Reconnect overflow fresh tracks back to older lost/removed tracks inside the fixed ID pool."""
        max_id_pool = self._max_id_pool()
        if max_id_pool <= 0:
            return

        new_tracks = [
            track for track in self.tracked_stracks
            if track.track_id > max_id_pool
            and 3 <= self._track_age(track) <= 100
        ]
        if not new_tracks:
            return

        active_ids = {track.track_id for track in self.tracked_stracks}
        unmatched_tracks = [
            track for track in self._reconnection_candidates(active_ids)
            if 1 <= track.track_id <= max_id_pool
        ]
        if not unmatched_tracks:
            return

        reconnection_distance_threshold = self.runtime_cfg.get("RECONNECTION_DISTANCE_THRESHOLD", 200)
        cost_matrix = np.full((len(unmatched_tracks), len(new_tracks)), np.inf, dtype=float)

        for i, unmatched_track in enumerate(unmatched_tracks):
            unmatched_center = self._track_last_detection_center(unmatched_track)
            for j, new_track in enumerate(new_tracks):
                new_center = self._track_first_detection_center(new_track)
                cost_matrix[i, j] = np.linalg.norm(unmatched_center - new_center)

        matches, _, _ = matching.linear_assignment(cost_matrix, thresh=reconnection_distance_threshold)
        for row_index, col_index in matches:
            old_track = unmatched_tracks[int(row_index)]
            new_track = new_tracks[int(col_index)]
            old_track_id = old_track.track_id
            new_track_id = new_track.track_id

            new_track.track_id = old_track_id
            self._merge_position_history(old_track_id, new_track_id)
            new_track.start_frame = old_track.start_frame
            new_track.tracklet_len += getattr(old_track, "tracklet_len", 0)
            self.lost_stracks = [track for track in self.lost_stracks if track.track_id != old_track_id]
            self.removed_stracks = [track for track in self.removed_stracks if track.track_id != old_track_id]
            old_track.mark_removed()

    def _reconnection_candidates(self, active_ids):
        candidates = []
        seen_ids = set()
        for track in [*self.lost_stracks, *self.removed_stracks]:
            if track.track_id in seen_ids or track.track_id in active_ids:
                continue
            if track.state not in {TrackState.Lost, TrackState.Removed}:
                continue
            if self.frame_id - track.frame_id <= 1:
                continue
            candidates.append(track)
            seen_ids.add(track.track_id)
        return candidates

    def _track_age(self, track):
        return self.frame_id - track.start_frame + 1

    def _track_last_detection_center(self, track):
        return self._track_history_center(track, -1)

    def _track_first_detection_center(self, track):
        return self._track_history_center(track, 0)

    def _track_history_center(self, track, index):
        track_id = self._track_id(track)
        if track_id is not None:
            history = self.position_history.get(track_id)
            if history:
                return np.asarray(history[index], dtype=float)
        return np.asarray(track.xywh[:2], dtype=float)

    def _track_center(self, track):
        return self._track_last_detection_center(track)

    def _max_id_pool(self):
        return int(self.runtime_cfg.get("MAX_ID_POOL", 0))

    def _record_detection(self, track, detection):
        track_id = self._track_id(track)
        if track_id is None:
            return
        detected_xywh = np.asarray(detection.xywh, dtype=float)
        self._append_position_history(track_id, detected_xywh[:2])

    def _append_position_history(self, track_id, center_xy):
        history = self.position_history.get(track_id)
        if history is None:
            history = deque(maxlen=self.position_history_length)
            self.position_history[track_id] = history
        history.append(np.asarray(center_xy, dtype=float).copy())

    def _merge_position_history(self, old_track_id, new_track_id):
        old_track_id = int(old_track_id)
        new_track_id = int(new_track_id)
        if old_track_id == new_track_id:
            return

        old_history = list(self.position_history.get(old_track_id, ()))
        new_history = list(self.position_history.get(new_track_id, ()))
        if old_history or new_history:
            merged_history = deque(maxlen=self.position_history_length)
            merged_history.extend((old_history + new_history)[-self.position_history_length:])
            self.position_history[old_track_id] = merged_history
        self.position_history.pop(new_track_id, None)

    def _track_id(self, track):
        track_id = getattr(track, "track_id", track)
        if track_id is None:
            return None
        return int(track_id)

    def _filter_detections_by_region(self, results, feats=None):
        if len(results) == 0:
            return results, feats

        x_min, x_max, y_min, y_max = self.detection_region
        xywh = np.asarray(self._to_numpy(results.xywh), dtype=float)
        centers = xywh[:, :2]
        keep_mask = (
            (centers[:, 0] >= x_min)
            & (centers[:, 0] <= x_max)
            & (centers[:, 1] >= y_min)
            & (centers[:, 1] <= y_max)
        )
        keep_indices = np.flatnonzero(keep_mask).astype(int).tolist()
        if len(keep_indices) == len(results):
            return results, feats
        return results[keep_indices], self._slice_feats(feats, keep_indices)

    def _to_numpy(self, value):
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    def _apply_detection_optimization(self, results, img: np.ndarray | None = None, feats=None):
        selected_results = results
        selected_feats = feats
        self._update_detection_optimization_target(len(results))
        target_count = self.detection_optimization_target

        if (
            target_count is not None
            and len(results) > target_count
            and len(self.bbox_history) > 0
            and len(self.bbox_history[-1]) == target_count
        ):
            prev_points = np.array(
                [self._bbox_center(box) for box in self.bbox_history[-1]], dtype=float
            ).reshape(-1, 2)
            curr_xywh = np.asarray(self._to_numpy(results.xywh), dtype=float)
            curr_points = curr_xywh[:, :2]
            curr_scores = np.asarray(self._to_numpy(results.conf), dtype=float)
            x_min, x_max, y_min, y_max = self.detection_region

            _, _, _, selected_indices = select_points_by_distance_and_confidence(
                prev_points=prev_points,
                curr_points=curr_points,
                curr_scores=curr_scores,
                alpha_distance=0.7,
                alpha_score=0.3,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
            selected_indices = [int(i) for i in selected_indices]
            selected_results = results[selected_indices]
            selected_feats = self._slice_feats(feats, selected_indices)
        self._update_frame_memory(np.asarray(self._to_numpy(selected_results.xywh), dtype=float))

        return selected_results, selected_feats

    def _update_detection_optimization_target(self, detection_count):
        """Sample the detection count and update the target after a stable sample window."""
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

    def _update_frame_memory(self, xywh_boxes):
        if len(self.bbox_history) >= self.frame_memory_length:
            self.bbox_history.pop(0)
        self.bbox_history.append(xywh_boxes)

    def _bbox_center(self, xywh_box):
        return [float(xywh_box[0]), float(xywh_box[1])]

    def _image_x_max(self, results, img):
        if img is not None:
            return img.shape[1] - 1
        return results.orig_shape[1] - 1

    def _image_y_max(self, results, img):
        if img is not None:
            return img.shape[0] - 1
        return results.orig_shape[0] - 1

    def _slice_feats(self, feats, indices):
        if feats is None:
            return None
        try:
            return feats[indices]
        except TypeError:
            return [feats[i] for i in indices]


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
    if target_count == 0:
        return curr_points[:0], curr_scores[:0], [], np.empty(0, dtype=int)

    d_max = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
    diff = prev_points[:, None, :] - curr_points[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    norm_distances = distances / d_max
    cost_matrix = alpha_distance * norm_distances - alpha_score * curr_scores[None, :]

    assignment_thresh = float(np.max(cost_matrix)) + 1.0
    matches, _, _ = matching.linear_assignment(cost_matrix, thresh=assignment_thresh)
    matches = np.asarray(matches, dtype=int)
    if matches.size == 0:
        raise ValueError("linear assignment returned no matches.")
    matches = matches[np.argsort(matches[:, 0])]
    prev_indices = matches[:, 0]
    curr_indices = matches[:, 1]

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
