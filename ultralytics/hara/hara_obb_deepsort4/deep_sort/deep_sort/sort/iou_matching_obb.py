import numpy as np
import sys
import os


from ultralytics.hara.hara_obb_deepsort4.deep_sort.deep_sort.sort import linear_assignment_obb
from ultralytics.hara.hara_obb_deepsort4.deep_sort.deep_sort.sort.obb_utils import rotated_iou, xyxyxyxy_to_xywhr


def obb_iou_xywhr(bbox_xywhr, candidates_xywhr):
    """Compute rotated intersection over union for OBB using xywhr format.

    Parameters
    ----------
    bbox_xywhr : tuple or ndarray
        An oriented bounding box in xywhr format (cx, cy, w, h, angle).
    candidates_xywhr : ndarray
        A matrix of candidate oriented bounding boxes (one per row) in xywhr format.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.
    """
    # Ensure bbox_xywhr is a tuple of floats
    if isinstance(bbox_xywhr, np.ndarray):
        bbox_xywhr = tuple(float(x) for x in bbox_xywhr)
    elif not isinstance(bbox_xywhr, tuple):
        bbox_xywhr = tuple(bbox_xywhr)

    ious = []
    for candidate in candidates_xywhr:
        # Ensure candidate is a tuple of floats
        if isinstance(candidate, np.ndarray):
            candidate = tuple(float(x) for x in candidate)
        elif not isinstance(candidate, tuple):
            candidate = tuple(candidate)

        iou_val = rotated_iou(bbox_xywhr, candidate)
        ious.append(iou_val)
    return np.array(ious)


def obb_iou_cost(tracks, detections, track_indices=None,
                 detection_indices=None):
    """A rotated intersection over union distance metric for OBB using xywhr format.

    Parameters
    ----------
    tracks : List[Track]
        A list of tracks with OBB state.
    detections : List[OBBDetection]
        A list of OBB detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - rotated_iou(track_xywhr, detection_xywhr)`.
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment_obb.INFTY_COST
            continue

        # Get track's xywhr representation from state (first 5 dimensions of mean)
        track = tracks[track_idx]
        track_xywhr = track.mean[:5] if hasattr(track, 'mean') else track.to_xywhr()

        # Get detection xywhr representations
        candidates_xywhr = np.asarray([detections[i].to_xywhr() for i in detection_indices])

        # Calculate IOU costs
        cost_matrix[row, :] = 1. - obb_iou_xywhr(track_xywhr, candidates_xywhr)

    return cost_matrix


def iou_cost_fallback(tracks, detections, track_indices=None,
                      detection_indices=None):
    """Fallback IOU cost using axis-aligned bounding boxes for OBB detections.

    This is used when rotated IOU calculation fails or for performance reasons.

    Parameters
    ----------
    tracks : List[Track]
        A list of tracks.
    detections : List[OBBDetection]
        A list of OBB detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix using axis-aligned IOU as fallback.
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment_obb.INFTY_COST
            continue

        # Get axis-aligned bbox from track
        bbox_tlwh = tracks[track_idx].to_tlwh()

        # Calculate IOU using axis-aligned boxes
        for col, detection_idx in enumerate(detection_indices):
            det_tlwh = detections[detection_idx].to_tlwh_aligned()

            # Convert tlwh to xyxy
            x1_t, y1_t, w_t, h_t = bbox_tlwh
            x2_t, y2_t = x1_t + w_t, y1_t + h_t

            x1_d, y1_d, w_d, h_d = det_tlwh
            x2_d, y2_d = x1_d + w_d, y1_d + h_d

            # Calculate intersection
            inter_x1 = max(x1_t, x1_d)
            inter_y1 = max(y1_t, y1_d)
            inter_x2 = min(x2_t, x2_d)
            inter_y2 = min(y2_t, y2_d)

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area_t = w_t * h_t
            area_d = w_d * h_d
            union_area = area_t + area_d - inter_area

            iou = inter_area / union_area if union_area > 0 else 0.0
            cost_matrix[row, col] = 1.0 - iou

    return cost_matrix
