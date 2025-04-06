#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source /opt/ros/humble/setup.bash

cd $SCRIPT_DIR
cd ../simulator/wheelchair_unity
source ./install/setup.bash
./src/vehicle_simulator/mesh/unity/environment/Model.x86_64 &
ros2 launch vehicle_simulator system_unity.launch