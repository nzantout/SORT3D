#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source /opt/ros/humble/setup.bash

cd $SCRIPT_DIR
cd ../ai_module
source ./install/setup.bash
ros2 run language_planner language_query_publisher
