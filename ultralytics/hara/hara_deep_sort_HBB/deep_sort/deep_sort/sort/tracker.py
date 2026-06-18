# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


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
        测量与轨迹关联的距离度量
    max_age : int
        Maximum number of missed misses before a track is deleted.
        删除轨迹前的最大未命中数
    n_init : int
        Number of frames that a track remains in initialization phase.
        确认轨迹前的连续检测次数。如果前n_init帧内发生未命中，则将轨迹状态设置为Deleted
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3, csv_path=None, flush_interval=30):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        if kalman_filter is not None:
            self.kf = kalman_filter.KalmanFilter()
        else:
            from kalman_filter import  KalmanFilter
            self.kf = KalmanFilter()

        self.tracks = []   # 保存一个轨迹列表，用于保存一系列轨迹
        self._next_id = 1  # 下一个分配的轨迹id
        # CSV logging buffer and configuration
        self.csv_path = csv_path or "tracking_output.csv"
        self._csv_buffer = []
        self._frames_since_flush = 0
        self._flush_interval = flush_interval
        self.frame_id = 0  # Global frame counter

    def predict(self):
        """Propagate track state distributions one time step forward.
        将跟踪状态分布向前传播一步

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.
        执行测量更新和轨迹管理

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Increment global frame counter
        self.frame_id += 1

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        
        # 1. 针对匹配上的结果
        for track_idx, detection_idx in matches:
            # 更新tracks中相应的detection
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        
        # 2. 针对未匹配的track, 调用mark_missed进行标记
        # track失配时，若Tentative则删除；若update时间很久也删除
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # 3. 针对未匹配的detection， detection失配，进行初始化
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        
        # 得到最新的tracks列表，保存的是标记为Confirmed和Tentative的track
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            # 获取所有Confirmed状态的track id
            if not track.is_confirmed():
                continue
            features += track.features # 将Confirmed状态的track的features添加到features列表
            # 获取每个feature对应的track_id
            targets += [track.track_id for _ in track.features]
            track.features = []
        # 距离度量中的特征集更新
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

            # Get center and dimensions from last detected xyah
            if track.last_detected_xyah is not None:
                try:
                    # xyah format: x, y, aspect_ratio, height
                    x, y, a, h = float(track.last_detected_xyah[0]), float(track.last_detected_xyah[1]), \
                                 float(track.last_detected_xyah[2]), float(track.last_detected_xyah[3])
                    w = a * h  # width = aspect_ratio * height
                    confidence = float(track.last_confidence) if track.last_confidence is not None else None
                except (TypeError, IndexError):
                    continue
            else:
                continue

            # For HBB (horizontal bounding box), no rotation angle
            # CSV format: frame_id, track_id, center_x, center_y, width, height, confidence, detected
            self._csv_buffer.append((int(global_frame_id), tid, x, y, w, h, confidence, bool(detected)))



    def save_csv(self, file_path: str = None, force: bool = False):
        """Flush internal CSV buffer to disk.

        If file_path is None, use self.csv_path.
        """
        import os
        import csv

        if not self._csv_buffer and not force:
            return

        out_path = file_path or self.csv_path
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

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])

            cost_matrix = np.ones((len(targets), len(features)))
            if features[0] is not None:
                cost_matrix = self.metric.distance(features, targets)

            # 计算门控后的成本矩阵（代价矩阵）
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        # 区分开confirmed tracks和unconfirmed tracks
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        # 对确定态的轨迹进行级联匹配，得到匹配的tracks、不匹配的tracks、不匹配的detections
        # matching_cascade 根据特征将检测框匹配到确认的轨迹。
        # 传入门控后的成本矩阵
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.        
        # 将未确定态的轨迹和刚刚没有匹配上的轨迹组合为 iou_track_candidates 
        # 并进行基于IoU的匹配
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1] # 刚刚没有匹配上的轨迹
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1] # 并非刚刚没有匹配上的轨迹
        # 对级联匹配中还没有匹配成功的目标再进行IoU匹配
        # min_cost_matching 使用匈牙利算法解决线性分配问题。
        # 传入 iou_cost，尝试关联剩余的轨迹与未确认的轨迹。
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b # 组合两部分匹配 
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        new_track = Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature)
        # set last detected info on the newly created track for CSV logging
        try:
            new_track.last_detected_xyah = detection.to_xyah()
        except Exception:
            new_track.last_detected_xyah = None
        try:
            new_track.last_confidence = detection.confidence
        except Exception:
            new_track.last_confidence = None
        self.tracks.append(new_track)
        # hara change starts;
        if self._next_id >=16:
            # if next id is 16, it means currently we have 15 ids.
            print("16")
            # if the detection couldn't match any existing track, it means the track feature's similarity couldn't match the detection's feature.

        # hara change ends;
        self._next_id += 1
