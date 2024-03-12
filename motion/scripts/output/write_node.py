#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped

class InfoWriter(Node):
    def __init__(self):
        super().__init__("writer_node")
        self.ground_truth_pose_sub = self.create_subscription(PoseStamped, '/sim/pose', self.gt_callback, 10)
        self.localization_pose_sub = self.create_subscription(Point, 'localization_pose', self.localization_callback, 10)
        self.open_loop_pose_sub = self.create_subscription(Point, 'open_loop_pose', self.open_loop_callback, 10)

        self.location = "/home/abhyudit/ros2_ws/src/AA275-Final-Project/motion/scripts/output/"
        self.gt_file = open(f'{self.location}mcl_slam_v2_gt_pose.txt', 'w')
        self.local_file = open(f'{self.location}mcl_slam_v2_local_pose.txt', 'w')
        self.ol_file = open(f'{self.location}mcl_slam_v2_ol_pose.txt', 'w')

    def gt_callback(self, msg : PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        line = f'{time} {x} {y}\n'
        self.gt_file.write(line)
        self.gt_file.flush()

    def localization_callback(self, msg : Point):
        x = msg.x
        y = msg.y
        time = msg.z

        line = f'{time} {x} {y}\n'
        self.local_file.write(line)
        self.local_file.flush()

    def open_loop_callback(self, msg : Point):
        x = msg.x
        y = msg.y
        time = msg.z

        line = f'{time} {x} {y}\n'
        self.ol_file.write(line)
        self.ol_file.flush()
        
def main(args=None):
    rclpy.init(args=args)
    writer = InfoWriter()
    rclpy.spin(writer)
    rclpy.shutdown()

if __name__ == "__main__":
    main()