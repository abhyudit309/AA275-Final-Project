import numpy as np
import scipy.linalg

from . import turtlebot_model as tb


class Ekf(object):
    """
    Base class for EKF Localization.

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


class EkfLocalization(Ekf):
    """
    EKF Localization.
    """

    def __init__(self, x0, Sigma0, R, map_lines, tf_base_to_camera, g):
        """
        EkfLocalization constructor.

        Inputs:
                       x0: np.array[3,]  - initial belief mean.
                   Sigma0: np.array[3,3] - initial belief covariance.
                        R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
                map_lines: np.array[J,2] - J map lines in rows representing (alpha, r).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.map_lines = (
            map_lines  # Matrix of J map lines with (alpha, r) as rows
        )
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        super(self.__class__, self).__init__(x0, Sigma0, R)

    def transition_model(self, u, dt):
        """
        Turtlebot dynamics (unicycle model).
        """
        # Compute g, Gx, Gu using tb.compute_dynamics().
        g, Gx, Gu = tb.compute_dynamics(self.x, u, dt)

        return g, Gx, Gu

    def measurement_model(self, z_raw, Q_raw):
        """
        Assemble one joint measurement and covariance from the individual values
        corresponding to each matched line feature.
        """
        v_list, Q_list, H_list = self.compute_innovations(z_raw, Q_raw)
        if not v_list:
            print(
                "Scanner sees {} lines but can't associate them with any map entries.".format(
                    z_raw.shape[0]
                )
            )
            return None, None, None

        # Compute z, Q.
        z = np.array(v_list).flatten()
        Q = scipy.linalg.block_diag(*Q_list)
        H = np.concatenate(H_list)

        return z, Q, H

    def compute_innovations(self, z_raw, Q_raw):
        """
        Given lines extracted from the scanner data, tries to associate each one
        to the closest map entry measured by Mahalanobis distance.

        Inputs:
            z_raw: np.array[I,2]   - I lines extracted from scanner data in
                                     rows representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) row of z_raw.
        Outputs:
            v_list: [np.array[2,]]  - list of at most I innovation vectors
                                      (predicted map measurement - scanner measurement).
            Q_list: [np.array[2,2]] - list of covariance matrices of the
                                      innovation vectors (from scanner uncertainty).
            H_list: [np.array[2,3]] - list of Jacobians of the innovation
                                      vectors with respect to the belief mean self.x.
        """
        hs, Hs = self.compute_predicted_measurements()

        # Compute v_list, Q_list, H_list
        v_list = []
        Q_list = []
        H_list = []
        I = z_raw.shape[0]  # Number of observed lines
        J = hs.shape[0]  # Number of predicted lines
        for i in range(I):
            zi = z_raw[i, :]
            Qi = Q_raw[i]
            dij_min = np.inf
            for j in range(J):
                hj = hs[j, :]
                vij = np.array([self.angle_diff(zi[0], hj[0]), zi[1] - hj[1]])
                Hj = Hs[j]
                Sij = Hj @ self.Sigma @ Hj.T + Qi 
                dij = np.matmul(vij, np.matmul(np.linalg.inv(Sij), vij.reshape(2, 1)))[0]
                if dij < dij_min:
                    dij_min = dij
                    vij_min = vij
                    Hj_min = Hj
            if dij_min < self.g ** 2:
                v_list.append(vij_min)
                Q_list.append(Qi)
                H_list.append(Hj_min)

        return v_list, Q_list, H_list

    def compute_predicted_measurements(self):
        """
        Given a single map line in the world frame, outputs the line parameters
        in the scanner frame so it can be associated with the lines extracted
        from the scanner measurements.

        Input:
            None
        Outputs:
                 hs: np.array[J,2]  - J line parameters in the scanner (camera) frame.
            Hx_list: [np.array[2,3]] - list of Jacobians of h with respect to the belief mean self.x.
        """
        hs = np.zeros_like(self.map_lines)
        Hx_list = []
        for j in range(self.map_lines.shape[0]):
            # Compute h, Hx using tb.transform_line_to_scanner_frame() for the j'th map line.
            h, Hx = tb.transform_line_to_scanner_frame(self.map_lines[j, :], self.x, self.tf_base_to_camera)
            h, Hx = tb.normalize_line_parameters(h, Hx)
            hs[j, :] = h
            Hx_list.append(Hx)

        return hs, Hx_list
    
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