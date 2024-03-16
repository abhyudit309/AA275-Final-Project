#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    mc = LaunchConfiguration("mc")
    num_particles = LaunchConfiguration("num_particles")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("mc", default_value="false"),
        DeclareLaunchArgument("num_particles", default_value="1000"),

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

        # Localization node
        Node(
            executable="pf_slam_node_v2.py",
            package="motion",
            parameters=[{
                "use_sim_time": use_sim_time,
                "/mc": mc,
                "/num_particles": num_particles
            }]
        ),

        # Writer node
        # Node(
        #     executable="write_node.py",
        #     package="motion",
        #     parameters=[{"use_sim_time": use_sim_time}]
        # )
    ])