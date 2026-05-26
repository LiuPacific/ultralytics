# vim: expandtab:ts=4:sw=4
from .obb_utils import xywhr_to_xyxyxyxy
from collections import deque


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class TrackOBB:
    """
    A single target track with OBB state space `(cx, cy, w, h, angle)` and associated
    velocities, where `(cx, cy)` is the center of the oriented bounding box, `w` is the
    width, `h` is the height, and `angle` is the rotation angle.
    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

        # Store position history for the last 300 frames
        self.position_history = deque(maxlen=300)

    def to_tlwh(self):
        """Get current position in axis-aligned bounding box format `(top left x, top left y,
        width, height)` for compatibility.

        Returns
        -------
        ndarray
            The axis-aligned bounding box approximation.
        """
        # Extract center, width, height from OBB state
        cx, cy, w, h = self.mean[:4]
        # Convert to tlwh format (axis-aligned approximation)
        tl_x = cx - w/2
        tl_y = cy - h/2
        return [tl_x, tl_y, w, h]

    def to_xywhr(self):
        """Get current position in OBB format `(center_x, center_y, width, height, angle)`.

        Returns
        -------
        ndarray
            The OBB parameters.
        """
        return self.mean[:5].copy()

    def to_xyxyxyxy(self):
        """Get current position as OBB corner points `(x1,y1, x2,y2, x3,y3, x4,y4)`.

        Returns
        -------
        ndarray
            The OBB corner points.
        """
        xywhr = self.to_xywhr()
        return xywhr_to_xyxyxyxy(*xywhr)

    def to_tlbr(self):
        """Get current position in axis-aligned bounding box format `(min x, min y, max x, max y)`.

        Returns
        -------
        ndarray
            The axis-aligned bounding box.
        """
        tlwh = self.to_tlwh()
        return [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        """
        # Convert OBB detection to xywhr format for Kalman filter
        measurement = detection.to_xywhr()
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, measurement)
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        # Store current position in history
        self.position_history.append(self.to_xywhr())

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def get_position_history(self) -> list:
        """Get the list of previous positions (last 300 frames).

        Returns
        -------
        list
            List of xywhr positions from the last 300 frames.
        """
        return list(self.position_history)
