#!/bin/bash

export MISTRAL_API_KEY="YOUR API KEY HERE"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1

cd $SCRIPT_DIR
cd ../ai_module
source ./install/setup.bash
ros2 run language_planner language_planner_node --platform mecanum &
sleep 5

cd ../semantic_mapper
python -m semantic_mapping.mapping_ros2_node --config config/mapping_mecanum_real.yaml --captioner_batch_size 16