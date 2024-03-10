import numpy as np
import scipy.linalg
import scipy.stats 
import random

EPSILON_OMEGA = 1e-3

class ParticleFilter(object):
    """
    Base class for Monte Carlo localization and FastSLAM.

    Usage:
        pf = ParticleFilter(x0, R)
        while True:
            pf.transition_update(u, dt)
            pf.measurement_update(z, Q)
            localized_state = pf.x
    """

    def __init__(self, x0, R):
        """
        ParticleFilter constructor.

        Inputs:
            x0: np.array[M,3] - initial particle states.
             R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
        """
        self.M = x0.shape[0]  # Number of particles
        self.xs = x0  # Particle set [M x 3]
        self.ws = np.repeat(1. / self.M, self.M)  # Particle weights (initialize to uniform) [M]
        self.R = R  # Control noise covariance (corresponding to dt = 1 second) [2 x 2]

    @property
    def x(self):
        """
        Returns the particle with the maximum weight for visualization.

        Output:
            x: np.array[3,] - particle with the maximum weight.
        """
        # x = np.average(self.xs, axis=0, weights=self.ws)
        idx = self.ws == self.ws.max()
        x = self.xs[idx].mean(axis=0)
        th = self.xs[idx, 2]
        # th = x[2]
        x[2] = np.arctan2(np.sin(th).mean(), np.cos(th).mean())
        return x

    def transition_update(self, u, dt):
        """
        Performs the transition update step by updating self.xs.

        Inputs:
            u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Output:
            None - internal belief state (self.xs) should be updated.
        """
        # Update self.xs.
        us = np.random.multivariate_normal(u, self.R, self.M)
        self.xs = self.transition_model(us, dt)

    def transition_model(self, us, dt):
        """
        Propagates exact (nonlinear) state dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """
        raise NotImplementedError("transition_model must be overridden by a subclass of EKF")

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
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        raise NotImplementedError("measurement_update must be overridden by a subclass of EKF")

    def resample(self, xs, ws):
        """
        Resamples the particles according to the updated particle weights.

        Inputs:
            xs: np.array[M,3] - matrix of particle states.
            ws: np.array[M,]  - particle weights.

        Output:
            None - internal belief state (self.xs, self.ws) should be updated.
        """
        r = np.random.rand() / self.M

        # Update self.xs, self.ws.
        c = np.cumsum(ws)
        m = np.arange(0, self.M)
        u = c[-1] * (r + m / self.M)
        idx = np.searchsorted(c, u, side='left')  # indices corresponding to resampled states
        self.xs = xs[idx]
        self.ws = ws[idx]

    def measurement_model(self, z_raw, Q_raw):
        """
        Converts raw measurements into the relevant Gaussian form (e.g., a
        dimensionality reduction).

        Inputs:
            z_raw: np.array[I,2]   - I lines extracted from scanner data in
                                     rows representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) row of z_raw.
        Outputs:
            z: np.array[2I,]   - joint measurement mean.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        raise NotImplementedError("measurement_model must be overridden by a subclass of EKF")


class MonteCarloLocalization(ParticleFilter):

    def __init__(self, x0, R, tf_base_to_camera, g):
        """
        MonteCarloLocalization constructor.

        Inputs:
                       x0: np.array[M,3] - initial particle states.
                        R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        self.update = 0 # To expand the state
        self.feature_not_seen = np.empty(0) # Number of times a feature was NOT seen
        self.reject = 5 # Remove feature from state if not seen exceeds this
        self.P_map = np.empty(shape=(0, 0)) # Covariance for line parameters
        super(self.__class__, self).__init__(x0, R)

    def transition_model(self, us, dt):
        """
        Unicycle model dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """
        # Compute g.
        # indices corresponding to abs(om) <= EPSILON_OMEGA
        idx1 = np.where(np.abs(us[:, 1]) <= EPSILON_OMEGA)[0]
        x_new1 = self.xs[idx1, 0] + np.multiply(us[idx1, 0], np.cos(self.xs[idx1, 2])) * dt
        y_new1 = self.xs[idx1, 1] + np.multiply(us[idx1, 0], np.sin(self.xs[idx1, 2])) * dt
        theta_new1 = self.xs[idx1, 2] + us[idx1, 1] * dt
        g1 = np.vstack((x_new1, y_new1, theta_new1)).T

        # indices corresponding to abs(om) > EPSILON_OMEGA
        idx2 = np.where(np.abs(us[:, 1]) > EPSILON_OMEGA)[0]
        x_new2 = self.xs[idx2, 0] + np.multiply(np.divide(us[idx2, 0], us[idx2, 1]),
                                                 np.sin(self.xs[idx2, 2] + us[idx2, 1] * dt) -
                                                 np.sin(self.xs[idx2, 2]))
        y_new2 = self.xs[idx2, 1] - np.multiply(np.divide(us[idx2, 0], us[idx2, 1]),
                                                 np.cos(self.xs[idx2, 2] + us[idx2, 1] * dt) -
                                                 np.cos(self.xs[idx2, 2]))
        theta_new2 = self.xs[idx2, 2] + us[idx2, 1] * dt
        g2 = np.vstack((x_new2, y_new2, theta_new2)).T

        # final g matrix
        g = np.zeros((self.M, self.xs.shape[1]))
        J = (self.xs.shape[1] - 3) // 2
        if J > 0:
            u_map = np.random.multivariate_normal(np.zeros(2*J), self.P_map, self.M)
            g[:, 3:] = self.xs[:, 3:] + u_map
        g[idx1, :3] = g1
        g[idx2, :3] = g2

        return g

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
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        ws = np.zeros_like(self.ws)

        # Compute new particles (xs, ws) with updated measurement weights.
        vs, Q = self.measurement_model(z_raw, Q_raw)
        if vs is None:
            return
        ws = scipy.stats.multivariate_normal.pdf(vs, mean=None, cov=Q)
        xs = np.copy(self.xs)

        self.resample(xs, ws)

    def measurement_model(self, z_raw, Q_raw):
        """
        Assemble one joint measurement and covariance from the individual values
        corresponding to each matched line feature for each particle.

        Inputs:
            z_raw: np.array[I,2]   - I lines extracted from scanner data in
                                     rows representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) row of z_raw.
        Outputs:
            z: np.array[M,2I]  - joint measurement mean for M particles.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        if self.update % 50 == 0:
            for i in range(z_raw.shape[0]):
                # if random.uniform(0, 1) < 0.5:
                self.expand_state(z_raw[i, :], Q_raw[i])
        self.update += 1

        self.shrink_state()

        vs = self.compute_innovations(z_raw, np.array(Q_raw))

        # Compute Q.
        Q = scipy.linalg.block_diag(*Q_raw)

        return vs, Q

    def compute_innovations(self, z_raw, Q_raw):
        """
        Given lines extracted from the scanner data, tries to associate each one
        to the closest map entry measured by Mahalanobis distance.

        Inputs:
            z_raw: np.array[I,2]   - I lines extracted from scanner data in
                                     rows representing (alpha, r) in the scanner frame.
            Q_raw: np.array[I,2,2] - I covariance matrices corresponding
                                     to each (alpha, r) row of z_raw.
        Outputs:
            vs: np.array[M,2I] - M innovation vectors of size 2I
                                 (predicted map measurement - scanner measurement).
        """
        # Compute vs (with shape [M x I x 2]).
        hs = self.compute_predicted_measurements()
        I = z_raw.shape[0]  # Number of observed lines
        J = hs.shape[1]  # Number of predicted lines

        if J == 0:
            return None

        mat = np.zeros((self.M, I, J, 2))  # stores vij for each observation and each sample

        # compute difference between observed and predicted values
        alpha_diff = self.angle_diff(np.expand_dims(z_raw[:, 0], axis=(1, 2)),
                                np.expand_dims(hs[:, :, 0], axis=0))
        r_diff = np.expand_dims(z_raw[:, 1], axis=(1, 2)) - np.expand_dims(hs[:, :, 1], axis=0)
        mat[:, :, :, 0] = alpha_diff.transpose(1, 0, 2)
        mat[:, :, :, 1] = r_diff.transpose(1, 0, 2)

        # changes dimensions of Q_raw from I x 2 x 2 to M x I x 2 x 2
        Q_big = np.repeat(Q_raw[np.newaxis, :, :, :], self.M, axis=0)

        # 'maha_dist' stores the Mahalanobis distance on its diagonals for each
        # observation and each sample. Its dimensions are M x I x J x J
        maha_dist = np.matmul(mat, np.matmul(np.linalg.inv(Q_big), mat.transpose(0, 1, 3, 2)))
        diag_elements = np.diagonal(maha_dist, axis1=2, axis2=3)
        idx = np.argmin(diag_elements, axis=2)  # indices corresponding to minimum distance

        # creating arrays to index from 'mat'
        i1 = np.repeat(np.array(range(self.M)), I)
        i2 = np.repeat(np.array([range(I)]), self.M, axis=0).flatten()

        # indexing from 'mat' to get 'vs'
        vs = mat[i1, i2, idx.flatten(), :].reshape((self.M, I, 2))

        max_ws_idx = self.ws == self.ws.max()
        for j in range(J):
            if j in idx[max_ws_idx]:
                self.feature_not_seen[j] = 0
            else:
                self.feature_not_seen[j] += 1

        # Reshape [M x I x 2] array to [M x 2I]
        return vs.reshape((self.M,-1))  # [M x 2I]

    def compute_predicted_measurements(self):
        """
        Given a single map line in the world frame, outputs the line parameters
        in the scanner frame so it can be associated with the lines extracted
        from the scanner measurements.

        Input:
            None
        Output:
            hs: np.array[M,J,2] - J line parameters in the scanner (camera) frame for M particles.
        """
        # Compute hs.
        J = (self.xs.shape[1] - 3) // 2
        hs = np.zeros((self.M, J, 2))

        # pose of the camera in the world frame
        x_cam = self.xs[:, 0] + self.tf_base_to_camera[0] * np.cos(self.xs[:, 2]) - \
                self.tf_base_to_camera[1] * np.sin(self.xs[:, 2])
        y_cam = self.xs[:, 1] + self.tf_base_to_camera[0] * np.sin(self.xs[:, 2]) + \
                self.tf_base_to_camera[1] * np.cos(self.xs[:, 2])
        th_cam = self.xs[:, 2] + self.tf_base_to_camera[2]

        # broadcast 1D arrays to M x J matrices
        alpha_MJ = self.xs[:, 3 : 2 * J + 3 : 2]
        r_MJ = self.xs[:, 4 : 2 * J + 3 : 2]
        x_cam_MJ = np.tile(x_cam.reshape(self.M, 1), J)
        y_cam_MJ = np.tile(y_cam.reshape(self.M, 1), J)
        th_cam_MJ = np.tile(th_cam.reshape(self.M, 1), J)

        # line parameters in the camera frame
        alpha_in_cam = alpha_MJ - th_cam_MJ
        r_in_cam = r_MJ - np.multiply(x_cam_MJ, np.cos(alpha_MJ)) - \
                   np.multiply(y_cam_MJ, np.sin(alpha_MJ))

        # normalizing line parameters
        i1, i2 = np.where(r_in_cam < 0)
        r_in_cam[i1, i2] = -r_in_cam[i1, i2]
        alpha_in_cam[i1, i2] = np.pi + alpha_in_cam[i1, i2]
        alpha_in_cam = (alpha_in_cam + np.pi) % (2 * np.pi) - np.pi

        # final hs of dimensions M x J x 2
        hs[:, :, 0] = alpha_in_cam
        hs[:, :, 1] = r_in_cam

        return hs
    
    def expand_state(self, z, Q):
        z = z[np.newaxis, :]
        Q = Q[np.newaxis, :]
        hs = self.compute_predicted_measurements()
        I = 1  # Number of observed lines
        J = hs.shape[1]  # Number of predicted lines
        dij_min = np.inf

        if J > 0:
            mat = np.zeros((self.M, I, J, 2))  # stores vij for each observation and each sample

            # compute difference between observed and predicted values
            alpha_diff = self.angle_diff(np.expand_dims(z[:, 0], axis=(1, 2)),
                                    np.expand_dims(hs[:, :, 0], axis=0))
            r_diff = np.expand_dims(z[:, 1], axis=(1, 2)) - np.expand_dims(hs[:, :, 1], axis=0)
            mat[:, :, :, 0] = alpha_diff.transpose(1, 0, 2)
            mat[:, :, :, 1] = r_diff.transpose(1, 0, 2)

            # changes dimensions of Q_raw from I x 2 x 2 to M x I x 2 x 2
            Q_big = np.repeat(Q[np.newaxis, :, :, :], self.M, axis=0)

            # 'maha_dist' stores the Mahalanobis distance on its diagonals for each
            # observation and each sample. Its dimensions are M x I x J x J
            maha_dist = np.matmul(mat, np.matmul(np.linalg.inv(Q_big), mat.transpose(0, 1, 3, 2)))
            diag_elements = np.diagonal(maha_dist, axis1=2, axis2=3)
            diag_elements = diag_elements.reshape((self.M, J))

            max_ws_idx = self.ws == self.ws.max()
            dij_min = diag_elements[max_ws_idx].min()

        if dij_min > self.g ** 2:
            x_cam = self.xs[:, 0] + self.tf_base_to_camera[0] * np.cos(self.xs[:, 2]) - self.tf_base_to_camera[1] * np.sin(self.xs[:, 2])
            y_cam = self.xs[:, 1] + self.tf_base_to_camera[0] * np.sin(self.xs[:, 2]) + self.tf_base_to_camera[1] * np.cos(self.xs[:, 2])
            th_cam = self.xs[:, 2] + self.tf_base_to_camera[2]
            alpha_in_world = z[0][0] + th_cam
            r_in_world = z[0][1] + np.multiply(x_cam, np.cos(alpha_in_world)) + np.multiply(y_cam, np.sin(alpha_in_world))
            alpha_in_world = alpha_in_world.reshape((-1, 1))
            r_in_world = r_in_world.reshape((-1, 1))
            self.xs = np.hstack((self.xs, alpha_in_world, r_in_world))
            P = np.diag(np.array([0.001**2 , 0.002**2]))
            self.P_map = scipy.linalg.block_diag(self.P_map, P)
            self.feature_not_seen = np.append(self.feature_not_seen, 0)

    def shrink_state(self):
        remove_idx = np.where(self.feature_not_seen >= self.reject)[0]
        self.feature_not_seen = np.delete(self.feature_not_seen, remove_idx)
        idx_list = []
        idx_list_P = []
        for idx in remove_idx:
            idx_list.append(2*idx + 3)
            idx_list.append(2*idx + 4)
            idx_list_P.append(2*idx)
            idx_list_P.append(2*idx + 1)
        self.xs = np.delete(self.xs, idx_list, axis=1)
        self.P_map = np.delete(np.delete(self.P_map, idx_list_P, axis=0), idx_list_P, axis=1)
    
    def angle_diff(self, a, b):
        a = a % (2. * np.pi)
        b = b % (2. * np.pi)
        diff = a - b
        if np.size(diff) == 1:
            if np.abs(a - b) > np.pi:
                sign = 2. * (diff < 0.) - 1.
                diff += sign * 2. * np.pi
        else:
            idx = np.abs(diff) > np.pi
            sign = 2. * (diff[idx] < 0.) - 1.
            diff[idx] += sign * 2. * np.pi
        return diff