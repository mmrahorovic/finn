# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from enum import Enum
from typing import Any, List, Optional

from finn.transformation.fpgadataflow.vitis_build import VitisOptStrategy
from finn.util.basic import alveo_default_platform, alveo_part_map, pynq_part_map


class AutoFIFOSizingMethod(str, Enum):
    "Select the type of automatic FIFO sizing strategy."

    CHARACTERIZE = "characterize"
    LARGEFIFO_RTLSIM = "largefifo_rtlsim"


class ShellFlowType(str, Enum):
    """For builds that produce a bitfile, select the shell flow that will integrate
    the FINN-generated accelerator."""

    VIVADO_ZYNQ = "vivado_zynq"
    VITIS_ALVEO = "vitis_alveo"


class DataflowOutputType(str, Enum):
    "Output product types that can be generated by build_dataflow"

    STITCHED_IP = "stitched_ip"
    ESTIMATE_REPORTS = "estimate_reports"
    OOC_SYNTH = "out_of_context_synth"
    RTLSIM_PERFORMANCE = "rtlsim_performance"
    BITFILE = "bitfile"
    PYNQ_DRIVER = "pynq_driver"
    DEPLOYMENT_PACKAGE = "deployment_package"


class ComputeEngineMemMode(str, Enum):
    """Memory mode for generated compute engines. See
    https://finn.readthedocs.io/en/latest/internals.html#matrixvectoractivation-mem-mode
    for more information."""

    CONST = "const"
    DECOUPLED = "decoupled"


class VitisOptStrategyCfg(str, Enum):
    """Vitis optimization strategy with serializable string enum values."""

    DEFAULT = "default"
    POWER = "power"
    PERFORMANCE = "performance"
    PERFORMANCE_BEST = "performance_best"
    SIZE = "size"
    BUILD_SPEED = "quick"


class LargeFIFOMemStyle(str, Enum):
    """Type of memory resource to use for large FIFOs."""

    AUTO = "auto"
    BRAM = "block"
    LUTRAM = "distributed"
    URAM = "ultra"


class VerificationStepType(str, Enum):
    "Steps at which FINN ONNX execution can be launched for verification."

    #: verify after step_qonnx_to_finn, using Python execution
    QONNX_TO_FINN_PYTHON = "finn_onnx_python"
    #: verify after step_tidy_up, using Python execution
    TIDY_UP_PYTHON = "initial_python"
    #: verify after step_streamline , using Python execution
    STREAMLINED_PYTHON = "streamlined_python"
    #: verify after step_apply_folding_config, using C++ for each HLS node
    FOLDED_HLS_CPPSIM = "folded_hls_cppsim"
    #: verify after step_create_stitched_ip, using stitched-ip Verilog
    STITCHED_IP_RTLSIM = "stitched_ip_rtlsim"


#: List of steps that will be run as part of the standard dataflow build, in the
#: specified order. Use the `steps` as part of build config to restrict which
#: steps will be run.
default_build_dataflow_steps = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    "step_streamline",
    "step_convert_to_hls",
    "step_specialize_layers",
    "step_create_dataflow_partition",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_specialize_to_rtl",
    "step_hls_codegen",
    "step_hls_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "step_measure_rtlsim_performance",
    "step_out_of_context_synthesis",
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package",
]

#: List of steps to run for an estimate-only (no synthesis) dataflow build
estimate_only_dataflow_steps = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    "step_streamline",
    "step_convert_to_hls",
    "step_specialize_layers",
    "step_create_dataflow_partition",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
]

#: List of steps to run for a dataflow build including HLS code generation, but
#: without any synthesis.
hls_codegen_dataflow_steps = estimate_only_dataflow_steps + ["step_hls_codegen"]


@dataclass_json
@dataclass
class DataflowBuildConfig:
    """Build configuration to be passed to the build_dataflow function. Can be
    serialized into or de-serialized from JSON files for persistence.
    See list of attributes below for more information on the build configuration.
    """

    #: Directory where the final build outputs will be written into
    output_dir: str

    #: Target clock frequency (in nanoseconds) for Vivado synthesis.
    #: e.g. synth_clk_period_ns=5.0 will target a 200 MHz clock.
    #: If hls_clk_period_ns is not specified it will default to this value.
    synth_clk_period_ns: float

    #: Which output(s) to generate from the build flow.  See documentation of
    #: DataflowOutputType for available options.
    generate_outputs: List[DataflowOutputType]

    #: (Optional) Path to configuration JSON file. May include parallelization,
    #: FIFO sizes, RAM and implementation style attributes and so on.
    #: If the parallelization attributes (PE, SIMD) are part of the config,
    #: this will override the automatically generated parallelization
    #: attributes inferred from target_fps (if any)
    #: Will be applied with :py:mod:`qonnx.transformation.general.ApplyConfig`
    folding_config_file: Optional[str] = None

    #: (Optional) Target inference performance in frames per second.
    #: Note that target may not be achievable due to specific layer constraints,
    #: or due to resource limitations of the FPGA.
    #: If parallelization attributes are specified as part of folding_config_file
    #: that will override the target_fps setting here.
    target_fps: Optional[int] = None

    #: (Optional) Use two-pass relaxation for folding, only relevant if target_fps
    #: is set. If enabled, parallelization will internally run a second time if the
    #: target cycles from the first pass could not be achieved, instead using the
    #: achievable target to obtain a balanced pipeline. If disabled, this can be
    #: useful for decreasing the latency (even though throughput won't increase).
    folding_two_pass_relaxation: Optional[bool] = True

    #: (Optional) At which steps the generated intermediate output model
    #: will be verified. See documentation of VerificationStepType for
    #: available options.
    verify_steps: Optional[List[VerificationStepType]] = None

    #: (Optional) Name of .npy file that will be used as the input for
    #: verification. Only required if verify_steps is not empty.
    verify_input_npy: Optional[str] = "input.npy"

    #: (Optional) Name of .npy file that will be used as the expected output for
    #: verification. Only required if verify_steps is not empty.
    verify_expected_output_npy: Optional[str] = "expected_output.npy"

    #: (Optional) Save full execution context for each of the verify_steps.
    #: By default, only the top-level graph output is saved.
    verify_save_full_context: Optional[bool] = False

    #: (Optional) Save .vcd waveforms from rtlsim under reports.
    #: By default, waveforms won't be saved.
    verify_save_rtlsim_waveforms: Optional[bool] = False

    #: (Optional) Run synthesis to generate a .dcp for the stitched-IP output product.
    #: This can make it easier to treat it as a standalone artifact without requiring
    #: the full list of layer IP build directories. By default, synthesis will not run.
    stitched_ip_gen_dcp: Optional[bool] = False

    #: Insert a signature node to the stitched-IP to read/write information
    #: to the design: e.g. Customer signature, application signature, version
    signature: Optional[List[int]] = None

    #: (Optional) Control the maximum width of the per-PE MVAU stream while
    #: exploring the parallelization attributes to reach target_fps
    #: Only relevant if target_fps is specified.
    #: Set this to a large value (e.g. 10000) if targeting full unfolding or
    #: very high performance.
    mvau_wwidth_max: Optional[int] = 36

    #: (Optional) Whether thresholding layers (which implement quantized
    #: activations in FINN) will be implemented as stand-alone HLS layers,
    #: instead of being part of MatrixVectorActivation layer. This gives larger
    #: flexibility, and makes it possible to have runtime-writable thresholds.
    standalone_thresholds: Optional[bool] = False

    #: (Optional) Whether optimizations that minimize the bit width of the
    #: weights and accumulator will be applied. Because this optimization relies
    #: on the the values of the weights, it will only be applied if runtime-
    #: writeable weights is not enabled.
    minimize_bit_width: Optional[bool] = True

    #: Target board, only needed for generating full bitfiles where the FINN
    #: design is integrated into a shell.
    #: e.g. "Pynq-Z1" or "U250"
    board: Optional[str] = None

    #: Target shell flow, only needed for generating full bitfiles where the FINN
    #: design is integrated into a shell. See documentation of ShellFlowType
    #: for options.
    shell_flow_type: Optional[ShellFlowType] = None

    #: Target Xilinx FPGA part. Only needed when board is not specified.
    #: e.g. "xc7z020clg400-1"
    fpga_part: Optional[str] = None

    #: Whether FIFO depths will be set automatically. Involves running stitched
    #: rtlsim and can take a long time.
    #: If set to False, the folding_config_file can be used to specify sizes
    #: for each FIFO.
    auto_fifo_depths: Optional[bool] = True

    #: Whether FIFO nodes with depth larger than 32768 will be split.
    #: Allow to configure very large FIFOs in the folding_config_file.
    split_large_fifos: Optional[bool] = False

    #: When `auto_fifo_depths = True`, select which method will be used for
    #: setting the FIFO sizes.
    auto_fifo_strategy: Optional[AutoFIFOSizingMethod] = AutoFIFOSizingMethod.LARGEFIFO_RTLSIM

    #: Avoid using C++ rtlsim for auto FIFO sizing and rtlsim throughput test
    #: if set to True, always using Python instead
    force_python_rtlsim: Optional[bool] = False

    #: Memory resource type for large FIFOs
    #: Only relevant when `auto_fifo_depths = True`
    large_fifo_mem_style: Optional[LargeFIFOMemStyle] = LargeFIFOMemStyle.AUTO

    #: Target clock frequency (in nanoseconds) for Vivado HLS synthesis.
    #: e.g. `hls_clk_period_ns=5.0` will target a 200 MHz clock.
    #: If not specified it will default to synth_clk_period_ns
    hls_clk_period_ns: Optional[float] = None

    #: Which memory mode will be used for compute layers
    default_mem_mode: Optional[ComputeEngineMemMode] = ComputeEngineMemMode.DECOUPLED

    #: Force inference of RTL ConvolutionInputGenerator over HLS implementation
    #: If set to False, falls back to the default behavior of InferConvInpGen()
    force_rtl_conv_inp_gen: Optional[bool] = False

    #: Which Vitis platform will be used.
    #: Only relevant when `shell_flow_type = ShellFlowType.VITIS_ALVEO`
    #: e.g. "xilinx_u250_xdma_201830_2"
    #: If not specified but "board" is specified, will use the FINN
    #: default (if any) for that Alveo board
    vitis_platform: Optional[str] = None

    #: Path to JSON config file assigning each layer to an SLR.
    #: Only relevant when `shell_flow_type = ShellFlowType.VITIS_ALVEO`
    #: Will be applied with :py:mod:`qonnx.transformation.general.ApplyConfig`
    vitis_floorplan_file: Optional[str] = None

    #: Vitis optimization strategy
    #: Only relevant when `shell_flow_type = ShellFlowType.VITIS_ALVEO`
    vitis_opt_strategy: Optional[VitisOptStrategyCfg] = VitisOptStrategyCfg.DEFAULT

    #: Whether intermediate ONNX files will be saved during the build process.
    #: These can be useful for debugging if the build fails.
    save_intermediate_models: Optional[bool] = True

    #: Whether hardware debugging will be enabled (e.g. ILA cores inserted to
    #: debug signals in the generated hardware)
    enable_hw_debug: Optional[bool] = False

    #: Whether pdb postmortem debuggig will be launched when the build fails
    enable_build_pdb_debug: Optional[bool] = True

    #: When True, all warnings and compiler output will be printed in stdout.
    #: Otherwise, these will be suppressed and only appear in the build log.
    verbose: Optional[bool] = False

    #: If given, only run the steps in the list. If not, run default steps.
    #: See `default_build_dataflow_steps` for the default list of steps.
    #: When specified:
    #: Each item can either be a string, or a function (does not apply to json
    #: serialized configs) and does the following:
    #: - strings are resolved to functions from the default list
    #: - functions are called with (model, DataflowBuildConfig) as args
    steps: Optional[List[Any]] = None

    #: If given, start from this step, loading the intermediate model generated
    #: from the previous step (save_intermediate_models must be enabled)
    start_step: Optional[str] = None

    #: If given, stop at this step.
    stop_step: Optional[str] = None

    #: The optional argument `max_multithreshold_bit_width` affects which Quant nodes
    #: of the QONNX format get converted to the MultiThreshold nodes of FINN. This
    #: only affects Quant nodes in the activation path. Quant nodes, which define a
    #: bit width larger than `max_multithreshold_bit_width` are not converted to
    #: MultiThreshold nodes and a warning is raised instead.
    #: If not given `max_multithreshold_bit_width` defaults to 8.
    max_multithreshold_bit_width: Optional[int] = 8

    #: Override the number of inputs for rtlsim performance measurement.
    rtlsim_batch_size: Optional[int] = 1

    #: If set to True, FIFOs and DWCs with impl_style=vivado will be kept during
    #: rtlsim, otherwise they will be replaced by HLS implementations.
    rtlsim_use_vivado_comps: Optional[bool] = True

    def _resolve_hls_clk_period(self):
        if self.hls_clk_period_ns is None:
            # use same clk for synth and hls if not explicitly specified
            return self.synth_clk_period_ns
        else:
            return self.hls_clk_period_ns

    def _resolve_driver_platform(self):
        if self.shell_flow_type == ShellFlowType.VIVADO_ZYNQ:
            return "zynq-iodma"
        elif self.shell_flow_type == ShellFlowType.VITIS_ALVEO:
            return "alveo"
        else:
            raise Exception("Couldn't resolve driver platform for " + str(self.shell_flow_type))

    def _resolve_fpga_part(self):
        if self.fpga_part is None:
            # lookup from part map if not specified
            if self.shell_flow_type == ShellFlowType.VIVADO_ZYNQ:
                return pynq_part_map[self.board]
            elif self.shell_flow_type == ShellFlowType.VITIS_ALVEO:
                return alveo_part_map[self.board]
            else:
                raise Exception("Couldn't resolve fpga_part for " + self.board)
        else:
            # return as-is when explicitly specified
            return self.fpga_part

    def _resolve_cycles_per_frame(self):
        if self.target_fps is None:
            return None
        else:
            n_clock_cycles_per_sec = 10**9 / self.synth_clk_period_ns
            n_cycles_per_frame = n_clock_cycles_per_sec / self.target_fps
            return int(n_cycles_per_frame)

    def _resolve_vitis_opt_strategy(self):
        # convert human-readable enum to value expected by v++
        name_to_strategy = {
            VitisOptStrategyCfg.DEFAULT: VitisOptStrategy.DEFAULT,
            VitisOptStrategyCfg.POWER: VitisOptStrategy.POWER,
            VitisOptStrategyCfg.PERFORMANCE: VitisOptStrategy.PERFORMANCE,
            VitisOptStrategyCfg.PERFORMANCE_BEST: VitisOptStrategy.PERFORMANCE_BEST,
            VitisOptStrategyCfg.SIZE: VitisOptStrategy.SIZE,
            VitisOptStrategyCfg.BUILD_SPEED: VitisOptStrategy.BUILD_SPEED,
        }
        return name_to_strategy[self.vitis_opt_strategy]

    def _resolve_vitis_platform(self):
        if self.vitis_platform is not None:
            return self.vitis_platform
        elif (self.vitis_platform is None) and (self.board is not None):
            return alveo_default_platform[self.board]
        else:
            raise Exception(
                "Could not resolve Vitis platform:" " need either board or vitis_platform specified"
            )

    def _resolve_verification_steps(self):
        if self.verify_steps is None:
            return []
        else:
            return self.verify_steps

    def _resolve_verification_io_pair(self):
        if self.verify_steps is None:
            return None
        else:
            assert os.path.isfile(self.verify_input_npy), (
                "verify_input_npy not found: " + self.verify_input_npy
            )
            verify_input_npy = np.load(self.verify_input_npy)
            assert os.path.isfile(self.verify_expected_output_npy), (
                "verify_expected_output_npy not found: " + self.verify_expected_output_npy
            )
            verify_expected_output_npy = np.load(self.verify_expected_output_npy)
            return (
                verify_input_npy,
                verify_expected_output_npy,
            )
