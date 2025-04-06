#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source /opt/ros/noetic/setup.bash

cd $SCRIPT_DIR
cd ../simulator/wheelchair_unity
source ./devel/setup.bash
./src/vehicle_simulator/mesh/unity/environment/Model.x86_64 &
roslaunch vehicle_simulator system_unity.launch