#!/bin/bash

set -eux

CKPT_PATH=$(readlink -f $1)

cd $(dirname $0)/../

colcon build

set +eux
source install/setup.bash
set -eux

ros2 run mile_pkg mile_node --ros-args --param ckpt_path:=${CKPT_PATH}
