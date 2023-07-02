#!/bin/bash

set -eux

colcon build

set +eux
source install/setup.bash
set -eux

ros2 run mile mile_node
