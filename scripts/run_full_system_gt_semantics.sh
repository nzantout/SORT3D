#!/bin/bash

export MISTRAL_API_KEY="YOUR API KEY HERE"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source /opt/ros/humble/setup.bash

cd $SCRIPT_DIR
cd ../simulator/wheelchair_unity
source ./install/setup.bash
./src/vehicle_simulator/mesh/unity/environment/Model.x86_64 &
ros2 launch vehicle_simulator system_unity.launch &
sleep 5

cd $SCRIPT_DIR
cd ../ai_module_ros2
source ./install/setup.bash
ros2 launch language_planner sort3d_ros_launch.xml
