from __future__ import annotations

from typing import Any

import numpy as np

from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.bot_sort import BOTrack, BOTSORT
from ultralytics.trackers.byte_tracker import STrack
from ultralytics.trackers.utils import matching


class HaraBOTSORT(BOTSORT):
    """Hara-owned BOT-SORT tracker.

    This class is registered as a separate Ultralytics tracker type so Hara
    experiments can customize BOT-SORT behavior without editing the upstream
    `BOTSORT` implementation.
    """

    def __init__(self, args: Any):
        super().__init__(args)

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
        """Update tracker state.

        This local copy intentionally starts from Ultralytics BOT-SORT behavior,
        which uses the ByteTrack association flow with BOT-SORT-specific track
        initialization, distance, prediction, GMC, and optional ReID helpers.
        Customize this method for Hara-specific BOT-SORT experiments.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

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
        tracked_stracks: list[BOTrack] = []
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
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
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
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
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

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)
