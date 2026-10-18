import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("lidar_shape_classifier")
    params = os.path.join(share, "config", "classifier.yaml")
    return LaunchDescription(
        [
            Node(
                package="lidar_shape_classifier",
                executable="shape_classifier_node",
                name="shape_classifier",
                output="screen",
                parameters=[params],
            )
        ]
    )
