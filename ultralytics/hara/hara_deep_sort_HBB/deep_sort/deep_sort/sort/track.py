
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


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.

    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurence.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        # hits counts successful matches. Once it exceeds n_init, the state is
        # set to Confirmed. hits increments every time update() is called.
        self.hits = 1
        self.age = 1 # Number of frames elapsed since this track was created.
        # Incremented by predict() and reset to 0 by update().
        self.time_since_update = 0

        self.state = TrackState.Tentative # New tracks start in Tentative state.
        # Each track has multiple features; every update appends the latest feature.
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init 
        self._max_age = max_age

        self.position_history = deque(maxlen=300)

        # Last detection confidence is updated when a detection is associated.
        self.last_confidence = None


    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def get_detection_center(self):
        if self.position_history:
            return self.position_history[-1][:2].copy()
        return self.get_center()

    def get_center(self):
        ret = self.mean[:4].copy()
        return ret[:2]

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        measurement = detection.to_xyah()
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, measurement)
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        # hits counts successful matches. The track becomes Confirmed after
        # n_init consecutive matches.
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        # Store the associated detection measurement, not the Kalman-filtered state.
        self.position_history.append(measurement.copy())

        self.last_confidence = detection.confidence

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        # If a Tentative track does not match any detection, mark it Deleted.
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            # If time_since_update exceeds max_age, mark the track Deleted.
            # This happens after max_age consecutive missed matches.
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
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
            List of xyah detection measurements from the last 300 frames.
        """
        return list(self.position_history)

    def merge_with(self, new_track):
        """Merge an older track history into a newly created replacement track.

        This is used by tracker-level re-identification. The new track keeps its
        current Kalman state, but inherits the old track's historical context so
        downstream consumers continue to see one continuous identity.
        """
        new_track.position_history.extendleft(reversed(list(self.position_history)))
        # Prepend old appearance features to new track's features

        if self.hits:
            new_track.hits += self.hits
        if self.age:
            new_track.age += self.age
