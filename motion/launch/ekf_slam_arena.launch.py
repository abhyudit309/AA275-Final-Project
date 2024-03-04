#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),

        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare("asl_tb3_sim"), "launch", "rviz.launch.py"]),
            launch_arguments={
                "config": PathJoinSubstitution([
                    FindPackageShare("motion"),
                    "rviz",
                    "arena.rviz",
                ]),
                "use_sim_time": use_sim_time,
            }.items(),
        ),

        # student's SLAM node
        Node(
            executable="ekf_slam.py",
            package="motion",
            parameters=[{"use_sim_time": use_sim_time}]
        )
    ])
