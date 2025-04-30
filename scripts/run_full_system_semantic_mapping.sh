#!/bin/bash

export MISTRAL_API_KEY="YOUR API KEY HERE"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source /opt/ros/humble/setup.bash

cd $SCRIPT_DIR
source ../simulator/mecanum_unity/install/setup.bash
source ../ai_module/install/setup.bash

cd ../simulator/mecanum_unity
./src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model.x86_64 &
sleep 3 
ros2 launch vehicle_simulator system_simulation_with_route_planner.launch > /dev/null 2>&1 &
sleep 1
cd ../../ai_module
ros2 run rviz2 rviz2 -d src/language_planner/rviz/vehicle_simulator.rviz &
sleep 1

ros2 run language_planner language_planner_node --platform mecanum &
sleep 5

cd $SCRIPT_DIR/../semantic_mapper
python -m semantic_mapping.mapping_ros2_node --config config/mapping_mecanum_sim.yaml --captioner_batch_size 16