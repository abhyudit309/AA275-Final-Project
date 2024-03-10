import numpy as np
import scipy.linalg
import random

from . import turtlebot_model as tb


class Ekf(object):
    """
    Base class for EKF Localization and SLAM.

    Usage:
        ekf = EKF(x0, Sigma0, R)
        while True:
            ekf.transition_update(u, dt)
            ekf.measurement_update(z, Q)
            localized_state = ekf.x
    """

    def __init__(self, x0, Sigma0, R):
        """
        EKF constructor.

        Inputs:
                x0: np.array[n,]  - initial belief mean.
            Sigma0: np.array[n,n] - initial belief covariance.
                 R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
        """
        self.x = x0  # Gaussian belief mean
        self.Sigma = Sigma0  # Gaussian belief covariance
        self.R = R  # Control noise covariance (corresponding to dt = 1 second)

    def transition_update(self, u, dt):
        """
        Performs the transition update step by updating (self.x, self.Sigma).

        Inputs:
             u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Output:
            None - internal belief state (self.x, self.Sigma) should be updated.
        """
        g, Gx, Gu = self.transition_model(u, dt)

        # Update self.x, self.Sigma.
        self.x = g
        self.Sigma = Gx @ self.Sigma @ Gx.T + dt * Gu @ self.R @ Gu.T

    def transition_model(self, u, dt):
        """
        Propagates exact (nonlinear) state dynamics.

        Inputs:
             u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Outputs:
             g: np.array[n,]  - result of belief mean propagated according to the
                                system dynamics with control u for dt seconds.
            Gx: np.array[n,n] - Jacobian of g with respect to belief mean self.x.
            Gu: np.array[n,2] - Jacobian of g with respect to control u.
        """
        raise NotImplementedError(
            "transition_model must be overriden by a subclass of EKF"
        )

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[I,2]   - matrix of I rows containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) row of z_raw.
        Output:
            None - internal belief state (self.x, self.Sigma) should be updated.
        """
        z, Q, H = self.measurement_model(z_raw, Q_raw)
        if z is None:
            # Don't update if measurement is invalid
            # (e.g., no line matches for line-based EKF localization)
            return
        
        # Update self.x, self.Sigma.
        S = H @ self.Sigma @ H.T + Q
        K = self.Sigma @ H.T @ np.linalg.inv(S) 
        self.x = self.x + np.matmul(z, np.transpose(K))
        self.Sigma = self.Sigma - K @ S @ K.T

    def measurement_model(self, z_raw, Q_raw):
        """
        Converts raw measurements into the relevant Gaussian form (e.g., a
        dimensionality reduction). Also returns the associated Jacobian for EKF
        linearization.

        Inputs:
            z_raw: np.array[I,2]   - I lines extracted from scanner data in
                                     rows representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[2K,]   - measurement mean.
            Q: np.array[2K,2K] - measurement covariance.
            H: np.array[2K,n]  - Jacobian of z with respect to the belief mean self.x.
        """
        raise NotImplementedError(
            "measurement_model must be overriden by a subclass of EKF"
        )


class EkfSlam(Ekf):
    """
    EKF SLAM.
    """

    def __init__(self, x0, Sigma0, R, tf_base_to_camera, g):
        """
        EKFSLAM constructor.

        Inputs:
                       x0: np.array[3+2J,]     - initial belief mean.
                   Sigma0: np.array[3+2J,3+2J] - initial belief covariance.
                        R: np.array[2,2]       - control noise covariance
                                                 (corresponding to dt = 1 second).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        self.update = 0 # To expand the state
        self.feature_not_seen = np.empty(0) # Number of times a feature was NOT seen
        self.reject = 5 # Remove feature from state if not seen exceeds this
        super(self.__class__, self).__init__(x0, Sigma0, R)

    def transition_model(self, u, dt):
        """
        Combined Turtlebot + map dynamics.
        Adapt this method from EkfLocalization.transition_model().
        """
        g = np.copy(self.x)
        Gx = np.eye(self.x.size)
        Gu = np.zeros((self.x.size, 2))

        # Compute g, Gx, Gu.
        g_ekf, Gx_ekf, Gu_ekf = tb.compute_dynamics(self.x[0:3], u, dt)
        g[0:3] = g_ekf
        Gx[0:3, 0:3] = Gx_ekf
        Gu[0:3, 0:2] = Gu_ekf

        return g, Gx, Gu

    def measurement_model(self, z_raw, Q_raw):
        """
        Combined Turtlebot + map measurement model.
        Adapt this method from EkfLocalization.measurement_model().

        The ingredients for this model should look very similar to those for
        EkfLocalization. Instead of getting world-frame line parameters from 
        self.map_lines, you must extract them from the state self.x).

        We also periodically add features to expand the state, and drop features
        to shrink the state.
        """
        if self.update % 50 == 0:
            for i in range(z_raw.shape[0]):
                if random.uniform(0, 1) < 0.5:
                    self.expand_state(z_raw[i, :], Q_raw[i])
        self.update += 1

        self.shrink_state()
        
        v_list, Q_list, H_list = self.compute_innovations(z_raw, Q_raw)
        if not v_list:
            print(
                "Scanner sees {} lines but can't associate them with any map entries.".format(
                    z_raw.shape[0]
                )
            )
            return None, None, None

        # Compute z, Q, H.
        z = np.array(v_list).flatten()
        Q = scipy.linalg.block_diag(*Q_list)
        H = np.concatenate(H_list)

        return z, Q, H

    def compute_innovations(self, z_raw, Q_raw):
        """
        Adapt this method from EkfLocalization.compute_innovations().
        """
        hs, Hs = self.compute_predicted_measurements()

        # Compute v_list, Q_list, H_list.
        v_list = []
        Q_list = []
        H_list = []
        I = z_raw.shape[0]  # Number of observed lines
        J = hs.shape[0]  # Number of predicted lines
        js = []
        for i in range(I):
            zi = z_raw[i, :]
            Qi = Q_raw[i]
            dij_min = np.inf
            for j in range(J):
                hj = hs[j, :]
                vij = np.array([self.angle_diff(zi[0], hj[0]), zi[1] - hj[1]])
                Hj = Hs[j]
                Sij = np.matmul(Hj, np.matmul(self.Sigma, np.transpose(Hj))) + Qi
                dij = np.matmul(vij, np.matmul(np.linalg.inv(Sij), vij.reshape(2, 1)))[0]
                if dij < dij_min:
                    dij_min = dij
                    vij_min = vij
                    Hj_min = Hj
                    j_min = j
            if dij_min < self.g ** 2:
                js.append(j_min)
                v_list.append(vij_min)
                Q_list.append(Qi)
                H_list.append(Hj_min)

        for j in range(J):
            if j in js:
                self.feature_not_seen[j] = 0
            else:
                self.feature_not_seen[j] += 1

        return v_list, Q_list, H_list

    def compute_predicted_measurements(self):
        """
        Adapt this method from EkfLocalization.compute_predicted_measurements().
        """
        J = (self.x.size - 3) // 2
        hs = np.zeros((J, 2))
        Hx_list = []
        for j in range(J):
            idx_j = 3 + 2 * j
            alpha, r = self.x[idx_j: idx_j + 2]

            Hx = np.zeros((2, self.x.size))

            # Compute h, Hx.
            line = np.array([alpha, r])
            h_ekf, Hx_ekf = tb.transform_line_to_scanner_frame(line, self.x[0:3], self.tf_base_to_camera)
            h = h_ekf
            Hx[:, 0:3] = Hx_ekf
            Hx[:, idx_j:idx_j + 2] = np.eye(2)
            Hx[1, idx_j] = self.x[0] * np.sin(alpha) - self.x[1] * np.cos(alpha) - \
                                self.tf_base_to_camera[0] * np.sin(self.x[2] - alpha) - \
                                self.tf_base_to_camera[1] * np.cos(self.x[2] - alpha)

            h, Hx = tb.normalize_line_parameters(h, Hx)
            hs[j, :] = h
            Hx_list.append(Hx)

        return hs, Hx_list
    
    def expand_state(self, z, Q):
        J = (self.x.size - 3) // 2 # number of lines in state

        hs, Hs = self.compute_predicted_measurements()
        dij_min = np.inf
        for j in range(J):
            hj = hs[j, :]
            vij = np.array([self.angle_diff(z[0], hj[0]), z[1] - hj[1]])
            Hj = Hs[j]
            Sij = np.matmul(Hj, np.matmul(self.Sigma, np.transpose(Hj))) + Q
            dij = np.matmul(vij, np.matmul(np.linalg.inv(Sij), vij.reshape(2, 1)))[0]
            if dij < dij_min:
                dij_second_min = dij_min
                dij_min = dij
        if dij_min > self.g ** 2:
        # if J == 0 or dij_second_min / dij_min > 3:
            zi_in_world = tb.transform_line_to_world_frame(z, self.x[0:3], self.tf_base_to_camera)
            self.x = np.concatenate((self.x, zi_in_world))
            P = np.diag(np.array([0.1**2 , 0.2**2]))
            self.Sigma = scipy.linalg.block_diag(self.Sigma, P)
            self.feature_not_seen = np.append(self.feature_not_seen, 0)

    def shrink_state(self):
        remove_idx = np.where(self.feature_not_seen >= self.reject)[0]
        self.feature_not_seen = np.delete(self.feature_not_seen, remove_idx)
        idx_list = []
        for idx in remove_idx:
            idx_list.append(2*idx + 3)
            idx_list.append(2*idx + 4)
        self.x = np.delete(self.x, idx_list)
        self.Sigma = np.delete(np.delete(self.Sigma, idx_list, axis=0), idx_list, axis=1)

    def angle_diff(self, a, b):
        a = a % (2.0 * np.pi)
        b = b % (2.0 * np.pi)
        diff = a - b
        if np.size(diff) == 1:
            if np.abs(a - b) > np.pi:
                sign = 2.0 * (diff < 0.0) - 1.0
                diff += sign * 2.0 * np.pi
        else:
            idx = np.abs(diff) > np.pi
            sign = 2.0 * (diff[idx] < 0.0) - 1.0
            diff[idx] += sign * 2.0 * np.pi
        return diff