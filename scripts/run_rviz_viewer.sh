#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1

cd $SCRIPT_DIR
cd ../ai_module
ros2 run rviz2 rviz2 -d src/language_planner/rviz/rosbag_semantic_mapping.rviz
