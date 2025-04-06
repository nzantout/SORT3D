#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source /opt/ros/noetic/setup.bash

cd $SCRIPT_DIR
cd ../ai_module
source ./devel/setup.bash
rosrun language_planner language_query_publisher.py
