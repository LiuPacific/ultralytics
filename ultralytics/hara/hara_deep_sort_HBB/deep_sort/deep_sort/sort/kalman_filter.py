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

'''
Kalman filtering has two phases:
(1) Predict the track position at the next time step.
(2) Update the predicted position using the detection.
'''
class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        # Choose motion and observation uncertainty relative to the current
        # state estimate (height). These weights control model uncertainty.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        

        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        # Translates slice objects to concatenation along the first axis
        mean = np.r_[mean_pos, mean_vel]

        # Initialize the mean vector (8D) and covariance matrix (8x8) from the measurement.
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        # Predict from the object's previous mean and covariance.
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
       
        # Initialize noise matrix Q; np.r_ concatenates arrays along the first axis.
        # motion_cov is the covariance matrix Qk for process noise W_k.
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # Update time state x' = Fx (1)
        # x is the track mean at t-1. F is the state transition matrix, and this
        # equation predicts x' at time t.
        # self._motion_mat is F_k, the state transform applied to x_{k-1}.
        mean = np.dot(self._motion_mat, mean)
        # Calculate error covariance P' = FPF^T+Q (2)
        # P is the track covariance at t-1. Q is the system noise matrix, which
        # represents system reliability and is usually initialized with small values.
        # This equation predicts P' at time t.
        # covariance is P_{k|k}, the posterior estimation error covariance matrix.
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        # In equation 4, R is the detector noise matrix. It is a 4x4 diagonal
        # matrix whose diagonal values are the noise terms for the center
        # coordinates and width/height. Width/height noise is usually set larger
        # than center-point noise. The equation projects P' into detection space
        # and then adds the noise matrix R.
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
            
        # R is the measurement noise covariance; initialize the noise matrix R.
        innovation_cov = np.diag(np.square(std))

        # Project the mean vector into detection space, i.e. Hx'.
        mean = np.dot(self._update_mat, mean)
        # Project the covariance matrix into detection space, i.e. HP'H^T.
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov # Equation (4).

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        # Project mean and covariance into detection space to obtain Hx' and S.
        projected_mean, projected_cov = self.project(mean, covariance)

        # Matrix factorization.
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        # Compute Kalman gain K; this solves equation (5).
        # The Kalman gain controls the importance of the estimation error.
        # Cholesky factorization speeds up solving K. Equation (5) contains S
        # inverse on the right side; when S is large, solving the inverse is
        # expensive, so the implementation rewrites the problem as AX = B.
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        # y = z - Hx' (3)
        # In equation 3, z is the detection mean vector without velocity terms:
        # z = [cx, cy, r, h]. H is the measurement matrix; it projects the track
        # mean vector x' into detection space and computes the detection-track
        # mean error.
        innovation = measurement - projected_mean

        # Updated mean vector x = x' + Ky (6).
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # Updated covariance matrix P = (I - KH)P' (7).
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
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
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
