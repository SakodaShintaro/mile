#!/bin/bash

set -eux

CKPT_PATH=$(readlink -f $1)
CONF_PATH=$(readlink -f $2)

cd $(dirname $0)/../

colcon build

set +eux
source install/setup.bash
set -eux

export PYTHONPATH=${PYTHONPATH}:${HOME}/work/mile/

ros2 run mile_pkg mile_node \
    --ros-args --param ckpt_path:=${CKPT_PATH} \
    --ros-args --param conf_path:=${CONF_PATH}
