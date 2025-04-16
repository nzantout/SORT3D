#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source /opt/ros/humble/setup.bash

cd $SCRIPT_DIR
cd ../simulator/mecanum_unity
source ./install/setup.bash
./src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model.x86_64 &
sleep 3 
ros2 launch vehicle_simulator system_simulation_with_route_planner.launch > /dev/null 2>&1 &
sleep 1
cd ../../ai_module
ros2 run rviz2 rviz2 -d src/language_planner/rviz/vehicle_simulator.rviz
sleep 1