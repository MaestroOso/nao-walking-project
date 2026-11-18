"""
deploy.launch.py — policy node (+ optionally naoqi_driver2 and a rosbag).

    ros2 launch nao_rl_bridge deploy.launch.py export_dir:=<exported/NAME> \
        nao_ip:=<robot ip> dry_run:=true delta_deg:=0.0 repetition:=0 \
        [start_driver:=true] [record_bag:=true]

dry_run defaults to TRUE: nothing reaches the motors until you pass
dry_run:=false explicitly. Then use `ros2 run nao_rl_bridge walk_operator`
in another terminal to start / finish / stop the repetition.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument("export_dir", description="exported policy folder"),
        DeclareLaunchArgument("nao_ip", default_value="", description="robot IP (default: robot.yaml)"),
        DeclareLaunchArgument("dry_run", default_value="true"),
        DeclareLaunchArgument("delta_deg", default_value="0.0"),
        DeclareLaunchArgument("repetition", default_value="0"),
        DeclareLaunchArgument("label", default_value=""),
        DeclareLaunchArgument("robot_config", default_value=PathJoinSubstitution(
            [FindPackageShare("nao_rl_bridge"), "config", "robot.yaml"])),
        DeclareLaunchArgument("start_driver", default_value="false",
                              description="also launch naoqi_driver2"),
        DeclareLaunchArgument("record_bag", default_value="false"),
        DeclareLaunchArgument("bag_dir", default_value="nao_walk_bag"),
    ]
    lc = LaunchConfiguration

    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare("naoqi_driver"), "launch", "naoqi_driver.launch.py"])),
        launch_arguments={"nao_ip": lc("nao_ip")}.items(),
        condition=IfCondition(lc("start_driver")))

    node = Node(
        package="nao_rl_bridge", executable="policy_node", name="nao_policy_node", output="screen",
        parameters=[{
            "export_dir": lc("export_dir"), "robot_config": lc("robot_config"),
            "dry_run": lc("dry_run"), "robot_ip": lc("nao_ip"), "label": lc("label"),
            "delta_deg": lc("delta_deg"), "repetition": lc("repetition"),
        }])

    bag = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-o", lc("bag_dir"), "/joint_states", "/imu/torso",
             "/joint_angles", "/walk/command", "/walk/observation", "/walk/action", "/walk/status"],
        output="screen", condition=IfCondition(lc("record_bag")))

    return LaunchDescription(args + [driver, node, bag])
