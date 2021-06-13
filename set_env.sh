#!/bin/bash

export VIVADO_PATH=/opt/apps/xilinx/Vivado/2020.1
source $VIVADO_PATH/settings64.sh

export VITIS_PATH=/opt/apps/xilinx/Vitis/2020.1
source $VITIS_PATH/settings64.sh

export PLATFORM_REPO_PATHS=/opt/xilinx/platforms

export XILINX_XRT=/opt/xilinx/xrt

export FINN_INST_NAME=finn_dev_mirza
#export FINN_HOST_BUILD_DIR=/tmp/$FINN_INST_NAME
export FINN_HOST_BUILD_DIR=/workspace/results
export FINN_BUILD_DIR=$FINN_HOST_BUILD_DIR
export FINN_ROOT=/workspace/finn

export VIVADO_IP_CACHE=$FINN_HOST_BUILD_DIR/vivado_ip_cache
export ALVEO_BOARD="U250"

export NUM_DEFAULT_WORKERS=10

pip install --user -e /workspace/brevitas
pip install --user -e /workspace/finn-base

