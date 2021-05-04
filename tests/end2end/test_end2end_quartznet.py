# Copyright (c) 2021, Xilinx
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
# * Neither the name of FINN nor the names of its
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

import time
import pytest
import numpy as np

from finn.custom_op.registry import getCustomOp
from finn.util.test import (
    load_test_checkpoint_or_skip
)
from finn.core.modelwrapper import ModelWrapper
from finn.core.datatype import DataType
from finn.util.basic import get_by_name

from finn.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from finn.transformation.general import (
    GiveUniqueNodeNames,
    GiveRandomTensorNames,
    GiveReadableTensorNames,
    GiveUniqueParameterTensors
)
from finn.transformation.batchnorm_to_affine import BatchNormToAffine
from finn.transformation.streamline.reorder import (
    MoveAddPastMul,
    MoveAddPastConv,
    MoveMulPastFork,
    MoveScalarMulPastConv,
    MoveMulPastDWConv,
    MoveLinearPastEltwiseAdd
)
from finn.transformation.streamline.collapse_repeated import(
    CollapseRepeatedAdd,
    CollapseRepeatedMul
)
from finn.transformation.streamline.absorb import(
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    Absorb1BitMulIntoConv
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.create_generic_partitions import PartitionFromDict
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.absorb import AbsorbTransposeIntoMultiThreshold
from finn.transformation.streamline.reorder import (
    MoveTransposePastMultiThreshold,
    MoveTransposePastJoinAdd,
    MoveTransposeBeforeFork
)
from finn.transformation.extend_partition import ExtendPartition
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources

build_dir = os.environ["FINN_BUILD_DIR"]
mem_mode = "decoupled"

def test_end2end_quartznet_tidy_and_change_shape_tensors():
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_export.onnx")

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveRandomTensorNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(GiveUniqueParameterTensors())

    # Convert to supported format
    model = model.transform(Change3DTo4DTensors())

    model.save(build_dir+"/end2end_quartznet_tidy.onnx")

def test_end2end_quartznet_streamline():
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_tidy.onnx")
    # Collapse BatchNorm to Add and Mul
    model = model.transform(BatchNormToAffine())
    # Group additions
    model = model.transform(MoveAddPastMul())
    model = model.transform(MoveAddPastConv())
    model = model.transform(MoveAddPastMul())
    # Group multiplications
    model = model.transform(MoveMulPastFork())
    model = model.transform(MoveScalarMulPastConv())
    model = model.transform(MoveMulPastDWConv())
    # Move Mul/Add past join node
    model = model.transform(MoveLinearPastEltwiseAdd())
    # Collapes additions & multiplications
    model = model.transform(CollapseRepeatedAdd())
    model = model.transform(CollapseRepeatedMul())
    # Absorb Add/Mul into multithreshold
    model = model.transform(AbsorbAddIntoMultiThreshold())
    model = model.transform(FactorOutMulSignMagnitude())
    model = model.transform(Absorb1BitMulIntoConv())
    model = model.transform(AbsorbMulIntoMultiThreshold())

    # Ensure thresholds are integers
    ## Add quantization annotation to ensure RoundAndClipThresholds works
    global_input_name = model.graph.input[0].name
    model.set_tensor_datatype(global_input_name, DataType.INT8)
    model = model.transform(InferDataTypes())
    model = model.transform(RoundAndClipThresholds())

    # Remove floating point scalar multiplication before argmax
    mul_nodes = [x for x in model.graph.node if (x.op_type=="Mul")]
    final_mul_node = mul_nodes[-1]
    input_mul = final_mul_node.input[0]
    node_after_mul = model.find_consumer(final_mul_node.output[0])
    node_after_mul.input[0] = input_mul
    model.graph.node.remove(final_mul_node)

    model.save(build_dir+"/end2end_quartznet_streamline.onnx")

def test_end2end_quartznet_lowering():
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_streamline.onnx")
    partitionings = {0: range(0, 3),
                    1: range(3, 75),
                    2: range(75, 147),
                    3: range(147, 219),
                    4: range(219, 291),
                    5: range(291, 363),
                    6: range(363, 375)}
    model = model.transform(PartitionFromDict(partitionings))

    for n in model.graph.node:
        path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
        model_partition = ModelWrapper(path_to_partition)
        # Lowering
        model_partition = model_partition.transform(LowerConvsToMatMul())
        # Absorb transpose nodes
        model_partition = model_partition.transform(AbsorbTransposeIntoMultiThreshold())
        # Reorder remaining transpose nodes
        model_partition = model_partition.transform(MoveTransposePastMultiThreshold())
        model_partition = model_partition.transform(MoveTransposePastJoinAdd())
        model_partition = model_partition.transform(MoveTransposeBeforeFork())

        model_partition.save(path_to_partition)

    model.save(build_dir+"/end2end_quartznet_lowered.onnx")

def test_end2end_quartznet_repartition():
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_lowered.onnx")
    partitionings = [{0: range(0,4), 1: range(4, 34), 2: range(34, 63), 3: range(63, 92)},
                     {4: range(4, 33), 5: range(33, 62), 6: range(62, 91)},
                     {7: range(7, 36), 8: range(36, 65), 9: range(65, 94)},
                     {10: range(10, 39), 11: range(39, 68), 12: range(68, 97)},
                     {13: range(13, 42), 14: range(42, 71), 15: range(71, 100)},
                     {16: range(16, 25)}
                    ]

    nodes = [n for n in model.graph.node]
    for ind, n in enumerate(nodes):
        if ind == 0:
            node_ind_to_unfold = [ind, ind+1] # unfold current and next node
        else:
            node_ind_to_unfold = [3*ind+2] # (+1 for initial nodes, +3 partitions, +1 for Transpose node)

        model = model.transform(ExtendPartition(node_ind_to_unfold))
        model = model.transform(AbsorbTransposeIntoMultiThreshold())

        if ind==0:
            model = model.transform(PartitionFromDict(partitionings[0], "/tmp/finn_dev_mirza/partitioning_final"))
        if ind==1:
            model = model.transform(PartitionFromDict(partitionings[1], "/tmp/finn_dev_mirza/partitioning_final"))
        if ind==2:
            model = model.transform(PartitionFromDict(partitionings[2], "/tmp/finn_dev_mirza/partitioning_final"))
        if ind==3:
            model = model.transform(PartitionFromDict(partitionings[3], "/tmp/finn_dev_mirza/partitioning_final"))
        if ind==4:
            model = model.transform(PartitionFromDict(partitionings[4], "/tmp/finn_dev_mirza/partitioning_final"))
        if ind==5:
            model = model.transform(PartitionFromDict(partitionings[5], "/tmp/finn_dev_mirza/partitioning_final"))
            break

    model.save(build+dir+"/end2end_quartznet_lowered_partitioned.onnx")

def test_end2end_quartznet_convert_to_hls_layers():
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_lowered_partitioned.onnx")
    model = model.transform(to_hls.InferConvInpGen(), make_deepcopy=False)
    model = model.transform(to_hls.InferVVAU(), make_deepcopy=False)
    model = model.transform(to_hls.InferQuantizedStreamingFCLayer(), make_deepcopy=False)

    for n in model.graph.node:
        if n.op_type=="GenericPartition":
            path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
            model_partition = ModelWrapper(path_to_partition)

            model_partition = model_partition.transform(to_hls.InferConvInpGen(), make_deepcopy=False)
            model_partition = model_partition.transform(to_hls.InferVVAU(), make_deepcopy=False)
            model_partition = model_partition.transform(to_hls.InferQuantizedStreamingFCLayer(mem_mode), make_deepcopy=False)
            model_partition = model_partition.transform(to_hls.InferThresholdingLayer(), make_deepcopy=False)
            model_partition = model_partition.transform(to_hls.InferAddStreamsLayer(), make_deepcopy=False)
            model_partition = model_partition.transform(to_hls.InferDuplicateStreamsLayer(), make_deepcopy=False)

            model_partition.save(path_to_partition)

    model.save(build_dir+"/end2end_quartznet_hls_layers.onnx")


def test_end2end_quartznet_folding():
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_hls_layers.onnx")

    for n_par in model.graph.node:
        if n_par.op_type=="GenericPartition":
            path_to_partition = get_by_name(n_par.attribute, "model", "name").s.decode('utf-8')
            model_partition = ModelWrapper(path_to_partition)
            for n in model_partition.graph.node:
                if n.op_type=="StreamingFCLayer_Batch":
                    # Initial:
                    # SIMD=1
                    # PE=1
                    inst = getCustomOp(n)
                    mh = get_by_name(n.attribute, "MH", "name").i
                    mw = get_by_name(n.attribute, "MW", "name").i
                    if mh==29: # Check if we are at final node (TODO: make generic...)
                        assert(mw%4==0)
                        mh = 1
                    else:
                        assert(mh%4==0 and mw%4==0)
                        mh = int(mh/4)
                    mw = int(mw/4)
                    inst.set_nodeattr("PE", mh) # mh % PE ==0
                    inst.set_nodeattr("SIMD", mw) # mw % SIMD ==0
                if n.op_type=="Vector_Vector_Activate_Batch":
                    # Initial: PE = IFM_CH
                    inst = getCustomOp(n)
                    ifc = get_by_name(n.attribute, "Channels", "name").i
                    assert(ifc%4==0)
                    ifc = int(ifc/4)
                    inst.set_nodeattr("PE", ifc) # CH % PE == 0
                if n.op_type=="Thresholding_Batch":
                    # Initial: PE = 1
                    inst = getCustomOp(n)
                    ifc = get_by_name(n.attribute, "NumChannels", "name").i
                    assert(ifc%4==0)
                    ifc = int(ifc/4)
                    inst.set_nodeattr("PE", ifc) # CH % PE == 0
                if n.op_type=="AddStreams_Batch":
                    # Initial: PE = 1
                    inst = getCustomOp(n)
                    ifc = get_by_name(n.attribute, "NumChannels", "name").i
                    assert(ifc%4==0)
                    ifc = int(ifc/4)
                    inst.set_nodeattr("PE", ifc) # CH % PE == 0
                if n.op_type=="DuplicateStreams_Batch":
                    # Initial: PE = 1
                    inst = getCustomOp(n)
                    ifc = get_by_name(n.attribute, "NumChannels", "name").i
                    assert(ifc%4==0)
                    ifc = int(ifc/4)
                    inst.set_nodeattr("PE", ifc) # CH % PE == 0
                if n.op_type=="FMPadding_Batch":
                    # SIMD = IFM_CH
                    inst = getCustomOp(n)
                    ifc = get_by_name(n.attribute, "NumChannels", "name").i
                    assert(ifc%4==0)
                    ifc = int(ifc/4)
                    inst.set_nodeattr("SIMD", ifc) # CH % PE == 0
                if n.op_type=="ConvolutionInputGenerator1D":
                    # SIMD = IFM_CH
                    inst = getCustomOp(n)
                    ifc = get_by_name(n.attribute, "IFMChannels", "name").i
                    assert(ifc%4==0)
                    ifc = int(ifc/4)
                    inst.set_nodeattr("SIMD", ifc) # CH % PE == 0
            model_partition.save(path_to_partition)

    model.save(build_dir+"/end2end_quartznet_folded.onnx")


def test_end2end_quartznet_cppsim():
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_folded.onnx")

    start = time.time()
    for n in model.graph.node:
        if n.op_type=="GenericPartition":
            path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
            model_partition = ModelWrapper(path_to_partition)
            model_partition = model_partition.transform(PrepareCppSim())
            model_partition = model_partition.transform(CompileCppSim())
            model_partition = model_partition.transform(SetExecMode("cppsim"))
            model_partition.save(path_to_partition)
    end = time.time()

    elapsed_time = end-start
    f = open(build_dir + "/end2end_quartznet_compile_time.txt", "w+")
    f.write("Execution time in seconds: " + str(elapsed_time))
    f.close()

    model.save(build_dir + "/end2end_quartznet_cppsim.onnx")

def test_end2end_quartznet_gen_hls_ip():
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_folded.onnx")

    start = time.time()
    for n in model.graph.node:
        if n.op_type=="GenericPartition":
            path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
            model_partition = ModelWrapper(path_to_partition)
            model_partition = model_partition.transform(PrepareIP(test_fpga_part, target_clk_ns))
            model_partition = model_partition.transform(HLSSynthIP())
            model_partition = model_partition.transform(ReplaceVerilogRelPaths())
            model_partition = model_partition.transform(AnnotateResources("hls"))
            model_partition.save(path_to_partition)
    end = time.time()

    elapsed_time = end - start
    f = open(build_dir + "/end2end_mobilenet_ipgen_time.txt", "w+")
    f.write("Execution time in seconds: " + str(elapsed_time))
    f.close()

    model.save(build_dir+"/end2end_quartznet_ipgen.onnx")

def test_end2end_quartznet_rtlsim():
