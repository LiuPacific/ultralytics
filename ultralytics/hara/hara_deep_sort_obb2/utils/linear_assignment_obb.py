# vim: expandtab:ts=4:sw=4
# from __future__ import absolute_import
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from .kalman_filter_obb import KalmanFilterOBB

INFTY_COST = 1e+5

# Extended chi2inv95 for OBB (5 degrees of freedom)
chi2inv95_obb = 11.070  # 95% quantile of chi-square distribution with 5 df


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
    gating_threshold = chi2inv95_obb if gating_dim == 5 else kf.chi2inv95.get(gating_dim, 9.4877)

    measurements = np.asarray(
        [detections[i].to_xywhr() for i in detection_indices])

    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix

