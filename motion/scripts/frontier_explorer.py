#!/usr/bin/env python3

import numpy as np
import rclpy
import typing as T
from rclpy.node import Node
from scipy.signal import convolve2d

from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotState

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool


class FrontierExplorer(Node):

    def __init__(self) -> None:
        super().__init__("frontier_explorer")

        self.nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.explore_sub = self.create_subscription(Bool, "/explore", self.explore_callback, 10)
        self.nav_success_sub = self.create_subscription(Bool, "/nav_success", self.explore, 10)

        self.state: T.Optional[TurtleBotState] = None
        self.occupancy: T.Optional[StochOccupancyGrid2D] = None
        self.exploring = True

    def explore(self, _) -> None:
        if not self.exploring or self.state is None or self.occupancy is None:
            return

        window_size = 13
        sum_filter = np.ones((window_size, window_size))

        is_unknown = (self.occupancy.probs < 0).astype(float)
        is_occupied = (self.occupancy.probs >= 0.5).astype(float)

        num_unknown = convolve2d(is_unknown, sum_filter, mode="same", fillvalue=1)
        num_occupied = convolve2d(is_occupied, sum_filter, mode="same")
        num_free = window_size**2 - num_unknown - num_occupied

        grid_x, grid_y = np.meshgrid(np.arange(self.occupancy.size_xy[0]),
                                     np.arange(self.occupancy.size_xy[1]))
        grid_xy = np.stack([grid_x, grid_y], axis=-1)
        valid_points = (num_unknown >= window_size**2 * .2) & \
                       (num_occupied == 0) & \
                       (num_free >= window_size**2 * .3)

        if np.sum(valid_points) == 0:
            self.exploring = False
            self.get_logger().info("Finished exploring")
            return

        grid_xy = grid_xy[valid_points]
        state_xy = self.occupancy.grid2state(grid_xy)
        current_state_xy = np.array([self.state.x, self.state.y])
        dists = np.linalg.norm(state_xy - current_state_xy, axis=-1)
        idx = np.argmin(dists)

        target_state = TurtleBotState(
            x=state_xy[idx, 0],
            y=state_xy[idx, 1],
            theta=0.,
        )
        self.nav_pub.publish(target_state)

    def explore_callback(self, msg: Bool) -> None:
        self.exploring = msg.data
        self.explore(None)

    def state_callback(self, msg: TurtleBotState) -> None:
        self.state = msg

    def map_callback(self, msg: OccupancyGrid) -> None:
        if not self.exploring or self.state is None:
            return

        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=7,
            probs=msg.data,
        )


if __name__ == "__main__":
    rclpy.init()
    node = FrontierExplorer()
    rclpy.spin(node)
    rclpy.shutdown()