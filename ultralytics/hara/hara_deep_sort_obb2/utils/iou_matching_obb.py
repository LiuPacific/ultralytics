# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
import sys
import os
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))
import ultralytics.hara.hara_obb_deepsort.utils.linear_assignment_obb
# from ultralytics.hara.hara_obb_deepsort.deep_sort.deep_sort.sort import linear_assignment
from .obb_utils import rotated_iou


def obb_iou(bbox, candidates):
    """Compute rotated intersection over union for OBB.

    Parameters
    ----------
    bbox : ndarray
        An oriented bounding box in format `(x1,y1, x2,y2, x3,y3, x4,y4)` or `(cx, cy, w, h, angle)`.
    candidates : ndarray
        A matrix of candidate oriented bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.
    """
    ious = []
    for candidate in candidates:
        iou_val = rotated_iou(bbox, candidate)
        ious.append(iou_val)
    return np.array(ious)


def obb_iou_cost(tracks, detections, track_indices=None,
                 detection_indices=None):
    """A rotated intersection over union distance metric for OBB.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
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
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - rotated_iou(tracks[track_indices[i]], detections[detection_indices[j]])`.
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

        # Get OBB representation from track
        # For now, use axis-aligned approximation as tracks may not store rotation
        bbox = tracks[track_idx].to_tlwh()  # This gives axis-aligned tlwh
        # Convert tlwh to xyxy for compatibility with rotated_iou
        x1, y1, w, h = bbox
        bbox_xyxy = np.array([x1, y1, x1 + w, y1 + h])

        candidates = np.asarray([detections[i].to_xyxyxyxy().flatten() for i in detection_indices])
        cost_matrix[row, :] = 1. - obb_iou(bbox_xyxy, candidates)
    return cost_matrix


def iou_cost_fallback(tracks, detections, track_indices=None,
                      detection_indices=None):
    """Fallback IOU cost using axis-aligned bounding boxes for OBB detections.

    This is used when rotated IOU calculation fails or for performance reasons.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
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

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].to_tlwh_aligned() for i in detection_indices])

        # Standard axis-aligned IOU calculation
        bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
        candidates_tl = candidates[:, :2]
        candidates_br = candidates[:, :2] + candidates[:, 2:]

        tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
                   np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
        br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
                   np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
        wh = np.maximum(0., br - tl)

        area_intersection = wh.prod(axis=1)
        area_bbox = bbox[2:].prod()
        area_candidates = candidates[:, 2:].prod(axis=1)
        ious = area_intersection / (area_bbox + area_candidates - area_intersection)

        cost_matrix[row, :] = 1. - ious

    return cost_matrix
