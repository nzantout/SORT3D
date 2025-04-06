#!/bin/bash

export MISTRAL_API_KEY="YOUR API KEY HERE"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source /opt/ros/noetic/setup.bash

cd $SCRIPT_DIR
cd ../ai_module
source ./devel/setup.bash
roslaunch language_planner sort3d_ros.launch
