from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    xacro_file = "/home/fainic/robot_ws/sim_gen_scripts/generated/multi_arm_scene.xacro"

    world = LaunchConfiguration("world")
    entity_name = LaunchConfiguration("entity_name")

    robot_description = Command(["xacro", " ", xacro_file])

    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py",
            ])
        ]),
        launch_arguments={
            "gz_args": ["-r ", world],
        }.items(),
    )

    # Spawn entity in gazebo
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic",
            "robot_description",
            "-name",
            "multi_arm_scene",
            "-allow_renaming",
        ],
        output="both",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "world",
            default_value="empty.sdf",
            description="Gazebo world file or world arguments passed to gz sim",
        ),
        DeclareLaunchArgument(
            "entity_name",
            default_value="multi_arm_scene",
            description="Name of the spawned entity in Gazebo",
        ),
        rsp_node,
        gazebo,
        spawn_entity,
    ])