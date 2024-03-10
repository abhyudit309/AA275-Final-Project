import numpy as np

EPSILON_OMEGA = 1e-3

def compute_Gx(xvec, u, dt):
    """
    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
    Outputs:
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
    """
    ########## Code starts here ##########
    # TODO: Compute Gx
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.

    _, Gx, _ = compute_dynamics(xvec, u, dt)
        
    ########## Code ends here ##########
    return Gx
    

def compute_Gu(xvec, u, dt):
    """
    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
    Outputs:
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute Gu
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.

    _, _, Gu = compute_dynamics(xvec, u, dt)
        
    ########## Code ends here ##########
    return Gu


def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.

    x, y, th = xvec
    V, om = u

    if abs(om) < EPSILON_OMEGA:
        g = np.array([x + V * np.cos(th) * dt,
                      y + V * np.sin(th) * dt,
                      th + om * dt])

        Gx = np.array([[1, 0, -V * np.sin(th) * dt],
                       [0, 1, V * np.cos(th) * dt],
                       [0, 0, 1]])

        Gu = np.array([[np.cos(th) * dt, -V * np.sin(th) * dt * dt],
                       [np.sin(th) * dt, V * np.cos(th) * dt * dt],
                       [0, dt]])
    else:
        x_new = x + (V / om) * (np.sin(th + om * dt) - np.sin(th))
        y_new = y - (V / om) * (np.cos(th + om * dt) - np.cos(th))
        theta_new = th + om * dt

        g = np.array([x_new, y_new, theta_new])

        Gx = np.array([[1, 0, (V / om) * (np.cos(th + om * dt) - np.cos(th))],
                       [0, 1, (V / om) * (np.sin(th + om * dt) - np.sin(th))],
                       [0, 0, 1]])

        Gu_col1 = np.array([[(1 / om) * (np.sin(th + om * dt) - np.sin(th))],
                            [(-1 / om) * (np.cos(th + om * dt) - np.cos(th))],
                            [0]])

        Gu_col2 = np.array([[(V * dt / om) * np.cos(th + om * dt) -
                             (V / om ** 2) * (np.sin(th + om * dt) - np.sin(th))],
                            [(V * dt / om) * np.sin(th + om * dt) +
                             (V / om ** 2) * (np.cos(th + om * dt) - np.cos(th))],
                            [dt]])

        Gu = np.hstack((Gu_col1, Gu_col2))

    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)

    # pose of the camera in the world frame
    x_cam = x[0] + tf_base_to_camera[0] * np.cos(x[2]) - tf_base_to_camera[1] * np.sin(x[2])
    y_cam = x[1] + tf_base_to_camera[0] * np.sin(x[2]) + tf_base_to_camera[1] * np.cos(x[2])
    th_cam = x[2] + tf_base_to_camera[2]

    # line parameters in the camera frame
    alpha_in_cam = alpha - th_cam
    r_in_cam = r - x_cam * np.cos(alpha) - y_cam * np.sin(alpha)
    h = np.array([alpha_in_cam, r_in_cam])

    # Jacobian H
    Hx_row1 = np.array([0, 0, -1])
    Hx_row2 = np.array([-np.cos(alpha), -np.sin(alpha),
                        tf_base_to_camera[0] * np.sin(x[2] - alpha) +
                        tf_base_to_camera[1] * np.cos(x[2] - alpha)])
    Hx = np.vstack((Hx_row1, Hx_row2))

    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx

def transform_line_to_world_frame(line, x, tf_base_to_camera):
    """
    Given a single map line in the camera frame, outputs the line parameters
    in the world frame so it can be associated with the lines present in the state.

    Input:
                     line: np.array[2,] - map line (alpha, r) in camera frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.

    Outputs:
         h: np.array[2,]  - line parameters in the world frame.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)

    # pose of the camera in the world frame
    x_cam = x[0] + tf_base_to_camera[0] * np.cos(x[2]) - tf_base_to_camera[1] * np.sin(x[2])
    y_cam = x[1] + tf_base_to_camera[0] * np.sin(x[2]) + tf_base_to_camera[1] * np.cos(x[2])
    th_cam = x[2] + tf_base_to_camera[2]

    # line parameters in the camera frame
    alpha_in_world = alpha + th_cam
    r_in_world = r + x_cam * np.cos(alpha_in_world) + y_cam * np.sin(alpha_in_world)
    h = np.array([alpha_in_world, r_in_world])

    ########## Code ends here ##########

    return h


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
