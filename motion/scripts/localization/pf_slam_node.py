#!/usr/bin/env python3

# Runs PF SLAM with known correspondences

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.time import Time
from builtin_interfaces.msg import Time as Time_msg

from sensor_msgs.msg import LaserScan, PointCloud, ChannelFloat32
from geometry_msgs.msg import Twist, TransformStamped, Point32, PoseStamped, Point
from visualization_msgs.msg import Marker

import numpy as np
import tf2_ros
from collections import deque
import threading

from asl_tb3_lib.tf_utils import quaternion_to_yaw, quaternion_from_euler
from utils.ekf import EkfLocalization
from utils.particle_filter_slam import MonteCarloLocalization
from utils.ExtractLines import ExtractLines
from utils.maze_sim_parameters import LineExtractionParams, NoiseParams, ArenaParams, ARENA

from copy import deepcopy

def convert_to_time(time_msg: Time_msg):
    return time_msg.sec + time_msg.nanosec * 1e-9

def create_transform_msg(translation, rotation, child_frame, base_frame, time):
    t = TransformStamped()
    t.header.stamp = time
    t.header.frame_id = base_frame
    t.child_frame_id = child_frame
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]
    t.transform.rotation.x = rotation[0]
    t.transform.rotation.y = rotation[1]
    t.transform.rotation.z = rotation[2]
    t.transform.rotation.w = rotation[3]
    return t

def line_endpoints_from_alpha_and_r(alpha, r, d = 10.0):
    return ((r*np.cos(alpha) - d*np.sin(alpha), r*np.sin(alpha) + d*np.cos(alpha)),
            (r*np.cos(alpha) + d*np.sin(alpha), r*np.sin(alpha) - d*np.cos(alpha)))

class LocalizationVisualizer(Node):
    def __init__(self):
        super().__init__('turtlebot_localization', automatically_declare_parameters_from_overrides=True)

        ## Get parameters
        self.num_particles = self.get_parameter_or('/num_particles', Parameter('num_particles', Parameter.Type.INTEGER, 100)).value
        self.get_logger().info("LocalizationParams:\n"
                               f"num_particles = {self.num_particles}")

        ## Initial state for EKF
        self.EKF = None
        self.EKF_time = None
        self.current_control = np.zeros(2)
        self.latest_pose = None
        self.latest_pose_time = None
        self.base_to_camera = None
        self.controls = deque()
        self.scans = deque()

        N_map_lines = ArenaParams.shape[1]
        self.x0_map = ArenaParams.T.flatten()
        self.x0_map = self.x0_map + np.vstack((NoiseParams["std_alpha"]*np.random.randn(N_map_lines), NoiseParams["std_r"]*np.random.randn(N_map_lines))).T.flatten()
        self.P_map = np.diag(np.array([[NoiseParams["std_alpha"]**2 for _ in range(N_map_lines)], [NoiseParams["std_r"]**2 for _ in range(N_map_lines)]]).T.flatten())
        
        ## Set up publishers and subscribers
        self.tfBroadcaster = tf2_ros.TransformBroadcaster(self)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.control_sub = self.create_subscription(Twist, '/cmd_vel', self.control_callback, 10)
        self.state_sub = self.create_subscription(PoseStamped, '/sim/pose', self.state_callback, 10)
        self.ground_truth_ct = 0

        self.ground_truth_map_pub = self.create_publisher(Marker, "ground_truth_map", 10)
        self.ground_truth_map_marker = Marker()
        self.ground_truth_map_marker.header.frame_id = "odom"
        self.ground_truth_map_marker.header.stamp = self.get_clock().now().to_msg()
        self.ground_truth_map_marker.ns = "ground_truth"
        self.ground_truth_map_marker.type = 5    # line list
        self.ground_truth_map_marker.pose.orientation.w = 1.0
        self.ground_truth_map_marker.scale.x = .025
        self.ground_truth_map_marker.scale.y = .025
        self.ground_truth_map_marker.scale.z = .025
        self.ground_truth_map_marker.color.r = 0.0
        self.ground_truth_map_marker.color.g = 1.0
        self.ground_truth_map_marker.color.b = 0.0
        self.ground_truth_map_marker.color.a = 1.0
        self.ground_truth_map_marker.lifetime = Duration(seconds=1000).to_msg()
        self.ground_truth_map_marker.points = sum([[Point(x=float(p1[0]), y=float(p1[1]), z=0.0),
                                                    Point(x=float(p2[0]), y=float(p2[1]), z=0.0)] for p1, p2 in ARENA], [])

        self.EKF_map_pub = self.create_publisher(Marker, "MCL_map", 10)
        self.EKF_map_marker = deepcopy(self.ground_truth_map_marker)
        self.EKF_map_marker.color.r = 1.0
        self.EKF_map_marker.color.g = 0.0
        self.EKF_map_marker.color.b = 0.0
        self.EKF_map_marker.color.a = 1.0

        self.particles_pub = self.create_publisher(PointCloud, 'particle_filter', 10)
        
        ## Use simulation time
        self.set_parameters([Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.process_rate = self.create_rate(10.0)

        ## Colocate the `ground_truth` and `base_footprint` frames for visualization purposes
        tf2_ros.StaticTransformBroadcaster(self).sendTransform(
            create_transform_msg((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), "ground_truth", "base_footprint", self.get_clock().now().to_msg())
        )

    def scan_callback(self, msg):
        if self.EKF:
            self.scans.append((msg.header.stamp,
                               np.array([i*msg.angle_increment + msg.angle_min for i in range(len(msg.ranges))]),
                               np.array(msg.ranges)))

    def control_callback(self, msg):
        if self.EKF:
            self.controls.append((self.get_clock().now().to_msg(), np.array([msg.linear.x, msg.angular.z])))

    def state_callback(self, msg):
        self.ground_truth_ct = self.ground_truth_ct + 1
        self.latest_pose_time = self.get_clock().now().to_msg()
        self.latest_pose = msg.pose
        self.tfBroadcaster.sendTransform(create_transform_msg(
            (self.latest_pose.position.x, self.latest_pose.position.y, 0.0),
            (self.latest_pose.orientation.x, self.latest_pose.orientation.y, self.latest_pose.orientation.z, self.latest_pose.orientation.w),
            "base_footprint", "odom", self.latest_pose_time)
        )

    def process(self):
        while self.get_clock().now().to_msg() == Time(seconds=0, nanoseconds=0):
            self.process_rate.sleep()

        ## Get transformation of camera frame with respect to the base frame
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer, self)
        while True:
            try:
                # notably camera_link and not camera_depth_frame below, not sure why
                self.raw_base_to_camera = self.tfBuffer.lookup_transform("base_footprint", "base_scan", Time()).transform
                break
            except (tf2_ros.ConnectivityException, tf2_ros.LookupException):
                self.process_rate.sleep()
        rotation = self.raw_base_to_camera.rotation
        translation = self.raw_base_to_camera.translation
        tf_theta = quaternion_to_yaw(rotation)
        self.base_to_camera = [translation.x,
                               translation.y,
                               tf_theta] 

    def run(self):
        run_rate = self.create_rate(100.0)

        particles = PointCloud()
        particles.header.stamp = self.get_clock().now().to_msg() # self.EKF_time
        particles.header.frame_id = "odom"
        particles.points = [Point32(x=0.0, y=0.0, z=0.0) for _ in range(self.num_particles)]
        particle_intensities = ChannelFloat32()
        particle_intensities.name = "intensity"
        particle_intensities.values = [0.0 for _ in range(self.num_particles)]
        particles.channels.append(particle_intensities)

        while not self.latest_pose:
            run_rate.sleep()

        x0 = np.array([self.latest_pose.position.x,
                       self.latest_pose.position.y,
                       quaternion_to_yaw(self.latest_pose.orientation)])
        self.EKF_time = self.latest_pose_time

        while not self.base_to_camera:
            run_rate.sleep()

        final_x0 = np.concatenate((x0, self.x0_map))
        x0s = np.tile(np.expand_dims(final_x0, 0), (self.num_particles, 1))
        self.EKF = MonteCarloLocalization(x0s, 10. * NoiseParams["R"], self.P_map, self.base_to_camera, NoiseParams["g"])
        self.OLC = EkfLocalization(x0, NoiseParams["Sigma0"], NoiseParams["R"], ArenaParams.T, self.base_to_camera, NoiseParams["g"])

        while True:
            if not self.scans:
                run_rate.sleep()
                continue

            self.EKF_map_marker.points = []
            J = int((self.EKF.x.size - 3) / 2)
            for j in range(J):
                p1, p2 = line_endpoints_from_alpha_and_r(self.EKF.x[3+2*j], self.EKF.x[3+2*j+1])
                self.EKF_map_marker.points.extend([Point(x=p1[0], y=p1[1], z=0.0), Point(x=p2[0], y=p2[1], z=0.0)])
            self.ground_truth_map_pub.publish(self.ground_truth_map_marker)
            self.EKF_map_pub.publish(self.EKF_map_marker)

            while self.controls and convert_to_time(self.controls[0][0]) <= convert_to_time(self.scans[0][0]):
                next_timestep, next_control = self.controls.popleft()
                if convert_to_time(next_timestep) < convert_to_time(self.EKF_time):    # guard against time weirdness (msgs out of order)
                    continue
                self.EKF.transition_update(self.current_control,
                                           convert_to_time(next_timestep) - convert_to_time(self.EKF_time))
                self.OLC.transition_update(self.current_control,
                                           convert_to_time(next_timestep) - convert_to_time(self.EKF_time))
                self.EKF_time, self.current_control = next_timestep, next_control
                label = "MCL"
                self.tfBroadcaster.sendTransform(create_transform_msg(
                    (self.EKF.x[0], self.EKF.x[1], 0.0),
                    quaternion_from_euler(0.0, 0.0, self.EKF.x[2]),
                    label, "odom", self.EKF_time)
                )
                self.tfBroadcaster.sendTransform(create_transform_msg(
                    (self.OLC.x[0], self.OLC.x[1], 0.0),
                    quaternion_from_euler(0.0, 0.0, self.OLC.x[2]),
                    "open_loop", "odom", self.EKF_time)
                )

                particles.header.stamp = self.EKF_time
                for m in range(self.num_particles):
                    x = self.EKF.xs[m]
                    w = self.EKF.ws[m]
                    particles.points[m].x = x[0]
                    particles.points[m].y = x[1]
                    particles.channels[0].values[m] = w
                self.particles_pub.publish(particles)

            scan_time, theta, rho = self.scans.popleft()
            if convert_to_time(scan_time) < convert_to_time(self.EKF_time):
                continue
            self.EKF.transition_update(self.current_control,
                                       convert_to_time(scan_time) - convert_to_time(self.EKF_time))
            self.OLC.transition_update(self.current_control,
                                       convert_to_time(scan_time) - convert_to_time(self.EKF_time))
            self.EKF_time = scan_time
            alpha, r, C_AR, _, _ = ExtractLines(theta, rho,
                                                LineExtractionParams,
                                                NoiseParams["var_theta"],
                                                NoiseParams["var_rho"])
            Z = np.vstack((alpha, r)).T
            self.EKF.measurement_update(Z, C_AR)

            particles.header.stamp = self.EKF_time
            for m in range(self.num_particles):
                x = self.EKF.xs[m]
                w = self.EKF.ws[m]
                particles.points[m].x = x[0]
                particles.points[m].y = x[1]
                particles.channels[0].values[m] = w
            self.particles_pub.publish(particles)

            while len(self.scans) > 1:    # keep only the last element in the queue, if we're falling behind
                self.scans.popleft()

def main(args=None):
    rclpy.init(args=args)
    vis = LocalizationVisualizer()
    process_thread = threading.Thread(target=vis.process)
    process_thread.start()
    run_thread = threading.Thread(target=vis.run)
    run_thread.start()
    rclpy.spin(vis)
    rclpy.shutdown()

if __name__ == '__main__':
    main()