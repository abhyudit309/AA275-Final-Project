#!/usr/bin/env python3

import numpy as np
import rclpy
import scipy
import typing as T

from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.grids import StochOccupancyGrid2D


class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        if x==self.x_init or x==self.x_goal:
            return True
        for dim in range(len(x)):
            if x[dim] < self.statespace_lo[dim]:
                return False
            if x[dim] > self.statespace_hi[dim]:
                return False
        if not self.occupancy.is_free(np.asarray(x)):
            return False
        return True
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        return np.linalg.norm(np.array(x1)-np.array(x2))
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        for dx1 in [-self.resolution, 0, self.resolution]:
            for dx2 in [-self.resolution, 0, self.resolution]:
                if dx1==0 and dx2==0:
                    # don't include itself
                    continue
                new_x = (x[0]+dx1,x[1]+dx2)
                if self.is_free(new_x):
                    neighbors.append(self.snap_to_grid(new_x))
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        while len(self.open_set)>0:
            current = self.find_best_est_cost_through()
            if current == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            self.open_set.remove(current)
            self.closed_set.add(current)
            for n in self.get_neighbors(current):
                if n in self.closed_set:
                    continue
                tentative_cost_to_arrive = self.cost_to_arrive[current] + self.distance(current,n)
                if n not in self.open_set:
                    self.open_set.add(n)
                elif tentative_cost_to_arrive >= self.cost_to_arrive[n]:
                    continue
                self.came_from[n] = current
                self.cost_to_arrive[n] = tentative_cost_to_arrive
                self.est_cost_through[n] = self.cost_to_arrive[n] + self.distance(n,self.x_goal)

        return False
        ########## Code ends here ##########


class Navigator(BaseNavigator):

    def __init__(self) -> None:
        super().__init__()
        # heading controller params
        self.kth = 2.0

        # trajectory tracking control params
        self.kp = 0.5           # proportional control gain
        self.kd = 1.5           # derivative control gain
        self.v_thresh = 1e-3    # velocity singularity threshold
        self.v_desired = 0.15   # desired velocity for trajectory plans
        self.v_prev = 0.
        self.t_prev = 0.

        # planning parameters
        self.spline_alpha = 0.05

    def compute_heading_control(self,
        state: TurtleBotState,
        goal: TurtleBotState
    ) -> TurtleBotControl:
        control = TurtleBotControl()

        err = wrap_angle(goal.theta - state.theta)
        control.omega = self.kth * err

        return control

    def compute_trajectory_tracking_control(self,
        state: TurtleBotState,
        plan: TrajectoryPlan,
        t: float,
    ) -> TurtleBotControl:
        # desired state
        x_d = scipy.interpolate.splev(t, plan.path_x_spline, der=0)
        y_d = scipy.interpolate.splev(t, plan.path_y_spline, der=0)
        xd_d = scipy.interpolate.splev(t, plan.path_x_spline, der=1)
        yd_d = scipy.interpolate.splev(t, plan.path_y_spline, der=1)
        xdd_d = scipy.interpolate.splev(t, plan.path_x_spline, der=2)
        ydd_d = scipy.interpolate.splev(t, plan.path_y_spline, der=2)

        # avoid singularity
        if abs(self.v_prev) < self.v_thresh:
            self.v_prev = self.v_thresh

        # current state
        x = state.x
        y = state.y
        theta = state.theta
        xd = self.v_prev * np.cos(theta)
        yd = self.v_prev * np.sin(theta)

        # compute virtual controls
        u = np.array([xdd_d + self.kp * (x_d - x) + self.kd * (xd_d - xd),
                      ydd_d + self.kp * (y_d - y) + self.kd * (yd_d - yd)])

        # compute real controls
        J = np.array([[np.cos(theta), -self.v_prev * np.sin(theta)],
                      [np.sin(theta), self.v_prev * np.cos(theta)]])
        a, om = np.linalg.solve(J, u)
        v = self.v_prev + a * (t - self.t_prev)

        # record velocity and timestep
        self.v_prev = v
        self.t_prev = t

        return TurtleBotControl(v=v, omega=om)

    def compute_trajectory_plan(self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy: StochOccupancyGrid2D,
        resolution: float,
        horizon: float,
    ) -> TrajectoryPlan:
        problem = AStar(
            statespace_lo=(state.x - horizon, state.y - horizon),
            statespace_hi=(state.x + horizon, state.y + horizon),
            x_init=(state.x, state.y),
            x_goal=(goal.x, goal.y),
            occupancy=occupancy,
            resolution=resolution,
        )

        # uncessful or path too short for smoothing
        if problem.solve() == False or len(problem.path) < 4:
            return None

        # reset tracking controller history
        self.v_prev = 0.
        self.t_prev = 0.

        # use heuristic to compute path time with desired velocity
        path = np.asarray(problem.path)
        dt = np.zeros(path.shape[0])
        dt[1:] = np.linalg.norm(path[1:] - path[:-1], axis=-1) / self.v_desired
        ts = np.cumsum(dt)

        self.get_logger().info(f"state: {state}")
        self.get_logger().info(f"plan_init: {path[0]}")

        return TrajectoryPlan(
            path=path,
            path_x_spline=scipy.interpolate.splrep(ts, path[:, 0], k=3, s=self.spline_alpha),
            path_y_spline=scipy.interpolate.splrep(ts, path[:, 1], k=3, s=self.spline_alpha),
            duration=ts[-1],
        )


if __name__ == "__main__":
    rclpy.init()
    node = Navigator()
    rclpy.spin(node)
    rclpy.shutdown()
