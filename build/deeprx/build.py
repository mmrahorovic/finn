import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import alveo_default_platform
from finn.builder.build_dataflow_steps import (
    step_create_dataflow_partition,
    step_target_fps_parallelization,
    step_generate_estimate_reports
)
from warnings import warn
import os

from custom_steps import (
    step_deeprx_tidy,
    step_deeprx_streamline,
    step_deeprx_convert_to_hls
)

model_name = "deeprx_w4a4"
board = "U250"
vitis_platform = alveo_default_platform[board]
synth_clk_period_ns = 4.0
target_fps = 300

deeprx_build_steps = [
    step_deeprx_tidy,
    step_deeprx_streamline,
    step_deeprx_convert_to_hls,
    #step_create_dataflow_partition,
    #step_target_fps_parallelization,
    #step_generate_estimate_reports
]

platforms_to_build = ["U250"]

model_file = "deeprx_export.onnx"

cfg = build_cfg.DataflowBuildConfig(
    steps=deeprx_build_steps,
    output_dir="output_deeprx",
    synth_clk_period_ns=synth_clk_period_ns,
    board=board,
    shell_flow_type=build_cfg.ShellFlowType.VITIS_ALVEO,
    vitis_platform=vitis_platform,
    # throughput parameters (auto-folding)
    mvau_wwidth_max = 36,
    target_fps = target_fps,
    # enable extra performance optimizations (physopt)
    vitis_opt_strategy=build_cfg.VitisOptStrategyCfg.PERFORMANCE_BEST,
    generate_outputs=[
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ],
)
build.build_dataflow_cfg(model_file, cfg)