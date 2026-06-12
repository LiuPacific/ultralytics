# vim: expandtab:ts=4:sw=4
# from __future__ import absolute_import
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from ultralytics.hara.hara_obb_deepsort4.deep_sort.deep_sort.sort.kalman_filter_obb import KalmanFilterOBB

INFTY_COST = 1e+5

# Extended chi2inv95 for OBB (5 degrees of freedom)
chi2inv95_obb = 11.070  # 95% quantile of chi-square distribution with 5 df
chi2inv70_obb = 6.0644  # 70% quantile of chi-square distribution with 5 df
chi2inv60_obb = 5.1319  # 60% quantile of chi-square distribution with 5 df
chi2inv50_obb = 4.3515  # 50% quantile of chi-square distribution with 5 df


def gate_cost_matrix_obb(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering for OBB.

    Parameters
    ----------
    kf : KalmanFilterOBB
        The OBB Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[TrackOBB]
        A list of predicted tracks at the current time step.
    detections : List[OBBDetection]
        A list of OBB detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.
    """
    gating_dim = 2 if only_position else 5  # 5 dimensions for OBB: cx, cy, w, h, angle
    # gating_threshold = chi2inv95_obb if gating_dim == 5 else kf.chi2inv95.get(gating_dim, 9.4877)
    # gating_threshold = chi2inv70_obb if gating_dim == 5 else kf.chi2inv70.get(gating_dim, 6.0644)
    # gating_threshold = chi2inv60_obb if gating_dim == 5 else kf.chi2inv60.get(gating_dim, 6.0644)
    gating_threshold = chi2inv50_obb if gating_dim == 5 else kf.chi2inv50.get(gating_dim, 6.0644)

    measurements = np.asarray(
        [detections[i].to_xywhr() for i in detection_indices])

    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        # TODO hara: take the gate away.
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix


# min_cost_matching 使用匈牙利算法解决线性分配问题。
# 传入 门控余弦距离成本 或 iou cost
def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.
        返回包含以下三个条目的元组：

        匹配的跟踪和检测的索引列表，
        不匹配的轨迹索引的列表，
        未匹配的检测索引的列表。
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    # 计算成本矩阵
    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    # 执行匈牙利算法，得到指派成功的索引对，行索引为tracks的索引，列索引为detections的索引
    row_indices, col_indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    # 找出未匹配的detections
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    # 找出未匹配的tracks
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    # 遍历匹配的(track, detection)索引对
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        # 如果相应的cost大于阈值max_distance，也视为未匹配成功
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
        距离度量：
        输入：一个轨迹和检测列表，以及一个N个轨迹索引和M个检测索引的列表。
        返回：NxM维的代价矩阵，其中元素(i，j)是给定轨迹索引中第i个轨迹与
        给定检测索引中第j个检测之间的关联成本。
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
        门控阈值。成本大于此值的关联将被忽略。
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
        级联深度应设置为最大轨迹寿命。
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
        当前时间步的预测轨迹列表。
    detections : List[detection.Detection]
        A list of detections at the current time step.
        当前时间步的检测列表。
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
        轨迹索引列表，用于将 cost_matrix中的行映射到tracks的
         轨迹（请参见上面的说明）。 默认为所有轨迹。
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.
        将 cost_matrix中的列映射到的检测索引列表
         detections中的检测（请参见上面的说明）。 默认为全部检测。

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    返回包含以下三个条目的元组：

    匹配的跟踪和检测的索引列表，
    不匹配的轨迹索引的列表，
    未匹配的检测索引的列表。

    """

    # 分配track_indices和detection_indices两个列表
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    # 初始化匹配集matches M ← ∅
    # 未匹配检测集unmatched_detections U ← D
    unmatched_detections = detection_indices
    matches = []
    # 由小到大依次对每个level的tracks做匹配
    for level in range(cascade_depth):
        # 如果没有detections，退出循环
        if len(unmatched_detections) == 0:  # No detections left
            break

        # 当前level的所有tracks索引
        # 步骤6：Select tracks by age
        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        # 如果当前level没有track，继续
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        # 步骤7：调用min_cost_matching函数进行匹配
        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        matches += matches_l # 步骤8
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))  # 步骤9
    return matches, unmatched_tracks, unmatched_detections
