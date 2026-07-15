from __future__ import annotations

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

    def __init__(self, args: Any):
        super().__init__(args)
        self.runtime_cfg = cfg.get("ByteTrack", {})
        self.bbox_history = []
        self.frame_memory_length = self.runtime_cfg.get("FRAME_MEMORY_LENGTH", 5)

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

        # hara detection optimization
        if self.runtime_cfg.get("USE_OPTIMIZATION", False) and self.runtime_cfg.get("DETECTION_OPTIMIZATION_ON", True):
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
            if self.runtime_cfg.get("TRACK_RECONNECTION_ON", True):
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
            unmatched_center = self._track_center(unmatched_track)
            for j, new_track in enumerate(new_tracks):
                new_center = self._track_center(new_track)
                cost_matrix[i, j] = np.linalg.norm(unmatched_center - new_center)

        matches, _, _ = matching.linear_assignment(cost_matrix, thresh=reconnection_distance_threshold)
        for row_index, col_index in matches:
            old_track = unmatched_tracks[int(row_index)]
            new_track = new_tracks[int(col_index)]
            old_track_id = old_track.track_id

            new_track.track_id = old_track_id
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

    def _track_center(self, track):
        last_detected_xywh = getattr(track, "last_detected_xywh", None)
        if last_detected_xywh is not None:
            return np.asarray(last_detected_xywh[:2], dtype=float)
        return np.asarray(track.xywh[:2], dtype=float)

    def _record_detection(self, track, detection):
        track.last_detected_xywh = np.asarray(detection.xywh, dtype=float).copy()

    def _apply_detection_optimization(self, results, img: np.ndarray | None = None, feats=None):
        selected_results = results
        selected_feats = feats
        max_id_pool = self.runtime_cfg.get("MAX_ID_POOL", 15)

        if (
            len(results) > max_id_pool
            and len(self.bbox_history) > 0
            and len(self.bbox_history[-1]) == max_id_pool
        ):
            prev_points = np.array([self._bbox_center(box) for box in self.bbox_history[-1]], dtype=float)
            curr_xywh = np.asarray(results.xywh, dtype=float)
            curr_points = curr_xywh[:, :2]
            curr_scores = np.asarray(results.conf, dtype=float)

            _, _, _, selected_indices = select_points_by_distance_and_confidence(
                prev_points=prev_points,
                curr_points=curr_points,
                curr_scores=curr_scores,
                target_count=max_id_pool,
                alpha_distance=0.7,
                alpha_score=0.3,
                x_max=self._image_x_max(results, img),
                y_max=self._image_y_max(results, img),
            )
            selected_indices = [int(i) for i in selected_indices]
            selected_results = results[selected_indices]
            selected_feats = self._slice_feats(feats, selected_indices)
        else:
            print(
                f"Current frame has {len(results)} detections, which is not more than {max_id_pool} "
                f"or no previous frame with {max_id_pool} detections to compare with. Skipping selection step."
            )
        self._update_frame_memory(np.asarray(selected_results.xywh, dtype=float))

        return selected_results, selected_feats

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
    target_count=15,
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

    if prev_points.shape != (target_count, 2):
        raise ValueError(f"prev_points must have shape ({target_count}, 2).")

    if curr_points.ndim != 2 or curr_points.shape[1] != 2:
        raise ValueError("curr_points must have shape (m, 2).")

    if curr_scores.shape[0] != curr_points.shape[0]:
        raise ValueError("curr_scores length must match curr_points length.")

    if curr_points.shape[0] < target_count:
        raise ValueError(f"curr_points must contain at least {target_count} points.")

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
