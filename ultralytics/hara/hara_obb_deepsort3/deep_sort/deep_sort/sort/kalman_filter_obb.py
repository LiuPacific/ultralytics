# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilterOBB(object):
    """
    Extended Kalman filter for tracking oriented bounding boxes (OBB) in image space.

    The 10-dimensional state space

        cx, cy, w, h, angle, vcx, vcy, vw, vh, vangle

    contains the bounding box center position (cx, cy), width w, height h,
    rotation angle, and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (cx, cy, w, h, angle) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self):
        ndim, dt = 5, 1.  # 5 dimensions for position/orientation: cx, cy, w, h, angle

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        self._std_weight_angle = 1. / 180  # Additional weight for angle uncertainty

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            OBB coordinates (cx, cy, w, h, angle) with center position (cx, cy),
            width w, height h, and rotation angle.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (10 dimensional) and covariance matrix (10x10
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Initialize covariance matrix with appropriate uncertainties
        std = [
            float(2 * self._std_weight_position * measurement[2]),  # cx uncertainty
            float(2 * self._std_weight_position * measurement[3]),  # cy uncertainty
            float(2 * self._std_weight_position * measurement[2]),  # w uncertainty
            float(2 * self._std_weight_position * measurement[3]),  # h uncertainty
            float(2 * self._std_weight_angle * 180),               # angle uncertainty (in degrees)
            float(10 * self._std_weight_velocity * measurement[2]), # vcx uncertainty
            float(10 * self._std_weight_velocity * measurement[3]), # vcy uncertainty
            float(10 * self._std_weight_velocity * measurement[2]), # vw uncertainty
            float(10 * self._std_weight_velocity * measurement[3]), # vh uncertainty
            float(10 * self._std_weight_angle * 180)                # vangle uncertainty
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 10 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 10x10 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state.
        """
        std_pos = [
            float(self._std_weight_position * mean[2]),  # cx
            float(self._std_weight_position * mean[3]),  # cy
            float(self._std_weight_position * mean[2]),  # w
            float(self._std_weight_position * mean[3]),  # h
            float(self._std_weight_angle * 180)          # angle
        ]
        std_vel = [
            float(self._std_weight_velocity * mean[2]),  # vcx
            float(self._std_weight_velocity * mean[3]),  # vcy
            float(self._std_weight_velocity * mean[2]),  # vw
            float(self._std_weight_velocity * mean[3]),  # vh
            float(self._std_weight_angle * 180)          # vangle
        ]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # Update time state x' = Fx
        mean = np.dot(self._motion_mat, mean)
        # Calculate error covariance P' = FPF^T + Q
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (10 dimensional array).
        covariance : ndarray
            The state's covariance matrix (10x10 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        std = [
            float(self._std_weight_position * mean[2]),  # cx
            float(self._std_weight_position * mean[3]),  # cy
            float(self._std_weight_position * mean[2]),  # w
            float(self._std_weight_position * mean[3]),  # h
            float(self._std_weight_angle * 180)          # angle
        ]

        innovation_cov = np.diag(np.square(std))

        # Project mean to measurement space
        mean = np.dot(self._update_mat, mean)
        # Project covariance to measurement space
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (10 dimensional).
        covariance : ndarray
            The state's covariance matrix (10x10 dimensional).
        measurement : ndarray
            The 5 dimensional measurement vector (cx, cy, w, h, angle).

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Matrix decomposition for Kalman gain calculation
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)

        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T

        # Innovation: measurement - projected_mean
        innovation = measurement - projected_mean

        # Handle angle wrapping (ensure angle difference is within -180 to 180 degrees)
        innovation[4] = ((innovation[4] + 180) % 360) - 180

        # Update mean and covariance
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 5 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (10 dimensional).
        covariance : ndarray
            Covariance of the state distribution (10x10 dimensional).
        measurements : ndarray
            An Nx5 dimensional matrix of N measurements, each in
            format (cx, cy, w, h, angle).
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean

        # Handle angle wrapping for full measurements
        if not only_position and measurements.shape[1] > 4:
            d[:, 4] = ((d[:, 4] + 180) % 360) - 180

        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
