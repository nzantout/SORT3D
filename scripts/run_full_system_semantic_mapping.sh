#!/bin/bash

export MISTRAL_API_KEY="YOUR API KEY HERE"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source /opt/ros/humble/setup.bash

cd $SCRIPT_DIR
source ../simulator/wheelchair_unity/install/setup.bash
source ../ai_module/install/setup.bash

cd ../simulator/wheelchair_unity
./src/vehicle_simulator/mesh/unity/environment/Model.x86_64 &
ros2 launch language_planner vehicle_simulator_semantic_mapping_launch.xml &
sleep 5

ros2 run language_planner language_planner_node --platform wheelchair &
sleep 5

cd $SCRIPT_DIR/../semantic_mapper
python -m semantic_mapping.mapping_ros2_node --config config/mapping_wheelchair.yaml --captioner_batch_size 16