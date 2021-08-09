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
import os
import json
import pathlib
import pytest
import numpy as np

import finn.core.onnx_exec as oxe
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.general import RemoveStaticGraphInputs
from finn.transformation.infer_shapes import InferShapes
#import brevitas_examples.speech_to_text as stt

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
    Absorb1BitMulIntoConv,
    AbsorbSignBiasIntoMultiThreshold
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
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.util.basic import alveo_part_map, alveo_default_platform
from finn.util.config import extract_model_config_to_json
from finn.transformation.fpgadataflow.set_fifo_depths import (
    InsertAndSetFIFODepths,
    RemoveShallowFIFOs
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.res_estimation import (
    res_estimation,
    res_estimation_complete
)
from finn.analysis.fpgadataflow.op_and_param_counts import (
    aggregate_dict_keys,
    op_and_param_counts
)
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.core.throughput_test import throughput_test_rtlsim
from copy import deepcopy
from finn.transformation.fpgadataflow.vitis_build import (
    VitisOptStrategy,
    VitisBuild
)
from finn.transformation.general import ApplyConfig

build_dir = os.environ["FINN_BUILD_DIR"]
mem_mode = "decoupled"
test_board = "U250"
test_platform = alveo_default_platform[test_board]
test_fpga_part = alveo_part_map[test_board]
#target_clk_ns = 6 # Fails timing requirement
target_clk_ns = 7


def verify_step(model, step_name):
    print("Running verification for: {} ...".format(step_name), end="", flush=True)

    if step_name=="brevitas_export":
        #quartznet_torch = stt.quant_quartznet_perchannelscaling_4b(export_mode=True)
        iname = model.graph.input[0].name
        oname = model.graph.output[0].name
        inp_val = np.load(build_dir+"/librispeech_data/input_sample_float_0.npy")
        #inp_val = np.reshape(inp_val, (inp_val.shape[0], inp_val.shape[1], inp_val.shape[2]))
        input_dict = {iname: inp_val}
        # Execute FINN-base simulation
        output_dict = oxe.execute_onnx(model, input_dict, return_full_exec_context=True)
        produced = output_dict[oname]
        produced = np.reshape(produced, np.shape(produced)+(1,))
        # Save QuartzNet export output, which will be used as reference for other transformations
        np.save(build_dir+"/librispeech_data/quartznet_export_output_full.npy", output_dict)
        np.save(build_dir+"/librispeech_data/quartznet_export_output.npy", produced)
        ## Execute Pytorch/Brevitas simulation
        #inp_val_torch = torch.from_numpy(inp_val).float()
        ## Do forward pass and compare output
        expected = np.load(build_dir+"/librispeech_data/output_sample_float_0.npy")
        #expected = quartznet_torch.forward(inp_val_torch).detach().numpy()
    elif step_name=="stitched_ip_rtlsim":
        # Execute first part (ONNX) of graph
        iname = model.graph.input[0].name
        inp_val = np.load(build_dir+"/librispeech_data/input_sample_float_0.npy")
        inp_val = np.reshape(inp_val, np.shape(inp_val)+(1,)) # Make 4D tensor
        input_dict = {iname: inp_val}
        # Change tensor annotation to FLOAT for simulation (must be kept INT otherwise InferDataTypes will annotate all tensors as FLOAT)
        first_conv_out = model.graph.node[0].output[0]
        for t in [iname, first_conv_out]:
            model.set_tensor_datatype(t, DataType.FLOAT32)
        # Find index of StreamingDataflowPartition node
        sdf_node = model.get_nodes_by_op_type("StreamingDataflowPartition")
        assert(len(sdf_node)==1), "Found more than 1 StreamingDataflowPartition node!"
        sdf_node_idx = model.get_node_index(sdf_node[0])
        output_dict = oxe.execute_onnx(model, input_dict, return_full_exec_context=True, end_node=model.graph.node[sdf_node_idx-1])
        # Execute IPstitched part of graph
        sdf_input = sdf_node[0].input[0]
        inp_val_sdf = output_dict[sdf_input]
        inst = getCustomOp(sdf_node[0])
        path_to_partition = inst.get_nodeattr("model")
        model_partition = ModelWrapper(path_to_partition)
        iname_sdf = model_partition.graph.input[0].name
        oname_sdf = model_partition.graph.output[0].name
        input_dict_sdf = {iname_sdf: inp_val_sdf}
        # Execute FINN simulation
        output_dict_sdf = oxe.execute_onnx(model_partition, input_dict_sdf, return_full_exec_context=True)
        produced = output_dict_sdf[oname_sdf]
        np.save(build_dir+"/librispeech_data/quartznet_"+step_name+".npy", output_dict_sdf)
        # Compare against golden output (QuartzNet exported model)
        expected = np.load(build_dir+"/librispeech_data/quartznet_export_output.npy")
    else:
        iname = model.graph.input[0].name
        oname = model.graph.output[0].name
        inp_val = np.load(build_dir+"/librispeech_data/input_sample_float_0.npy")
        inp_val = np.reshape(inp_val, np.shape(inp_val)+(1,)) # make 4D tensor
        input_dict = {iname: inp_val}
        # Change tensor annotation to FLOAT for simulation (must be kept INT otherwise InferDataTypes will annotate all tensors as FLOAT)
        first_conv_out = model.graph.node[0].output[0]
        for t in [iname, first_conv_out]:
            model.set_tensor_datatype(t, DataType.FLOAT32)
        # Execute FINN simulation
        output_dict = oxe.execute_onnx(model, input_dict, return_full_exec_context=True)
        produced = output_dict[oname]
        np.save(build_dir+"/librispeech_data/quartznet_"+step_name+".npy", output_dict)
        # Compare against golden output (QuartzNet exported model)
        expected = np.load(build_dir+"/librispeech_data/quartznet_export_output.npy")

    res = np.isclose(expected, produced, atol=1e-3).all() # basically the same as np.array_equal(expected, produced) for integer outputs
    res_to_str = {True: "SUCCESS", False: "FAIL"}
    res_str = res_to_str[res]
    print(res_str)


def test_end2end_quartznet_brevitas_export():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_quartznet_export_dev.onnx")
    verify_step(model, "brevitas_export")

#def test_end2end_quartznet_brevitas_export(verify=False):
#    finn_onnx = build_dir+"/end2end_quartznet_export_dev.onnx"
#    quartznet_torch = stt.quant_quartznet_perchannelscaling_4b(export_mode=True)
#    ishape = (1, 64, 256)
#    bo.export_finn_onnx(quartznet_torch, ishape, finn_onnx)
#    model = ModelWrapper(finn_onnx)
#    model = model.transform(InferShapes())
#    model = model.transform(FoldConstants())
#    model = model.transform(RemoveStaticGraphInputs())
#
#    model.save(finn_onnx)
#    if verify:
#        verify_step(model, "brevitas_export")


def test_end2end_quartznet_tidy_and_change_shape_tensors(verify=False):
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_export_dev.onnx")

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveRandomTensorNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(GiveUniqueParameterTensors())

    # Convert to supported format
    model = model.transform(Change3DTo4DTensors())

    model.save(build_dir+"/end2end_quartznet_tidy.onnx")
    if verify:
        verify_step(model, "tidy_and_change_shape_tensors")

def test_end2end_quartznet_streamline(verify=False):
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_tidy.onnx")

    # Absorb sign bias from export into MultiThreshold node
    model = model.transform(AbsorbSignBiasIntoMultiThreshold())
    # Collapse BatchNorm to Add and Mul
    model = model.transform(BatchNormToAffine())
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
    for n_mul in mul_nodes:
        input_mul = n_mul.input[0]
        node_after_mul = model.find_consumer(n_mul.output[0])
        node_after_mul.input[0] = input_mul
        model.graph.node.remove(n_mul)

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveRandomTensorNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(GiveUniqueParameterTensors())

    model.save(build_dir+"/end2end_quartznet_streamline.onnx")
    if verify:
        verify_step(model, "streamline")

def test_end2end_quartznet_lowering(verify=False):
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_streamline.onnx")
    partitionings = {1: range(2, 75),
                    2: range(75, 147),
                    3: range(147, 219),
                    4: range(219, 291),
                    5: range(291, 363),
                    6: range(363, 375)}
    model = model.transform(PartitionFromDict(partitionings, build_dir+"/partitioning_lowering"))

    for n in model.graph.node:
        if n.op_type=="GenericPartition":
            path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
            model_partition = ModelWrapper(path_to_partition)
            # Lowering
            model_partition = model_partition.transform(LowerConvsToMatMul())
            # Absorb transpose node
            model_partition = model_partition.transform(AbsorbTransposeIntoMultiThreshold())
            # Reorder remaining transpose nodes
            model_partition = model_partition.transform(MoveTransposePastMultiThreshold())
            model_partition = model_partition.transform(MoveTransposePastJoinAdd())
            model_partition = model_partition.transform(MoveTransposeBeforeFork())

            model_partition.save(path_to_partition)

    model.save(build_dir+"/end2end_quartznet_lowered.onnx")

def test_end2end_quartznet_repartition(verify=False):
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_lowered.onnx")
    partitionings = [{0: range(0,3), 1: range(3, 33), 2: range(33, 62), 3: range(62, 91), 4: range(91, 120), 5: range(120, 149), 6: range(149, 178)},
                     {7: range(7, 36), 8: range(36, 65), 9: range(65, 94)},
                     {10: range(10, 39), 11: range(39, 68), 12: range(68, 97)},
                     {13: range(13, 42), 14: range(42, 71), 15: range(71, 100)},
                     {16: range(16, 25)}
                    ]

    nodes = [n for n in model.graph.node if n.op_type=="GenericPartition"]
    for ind, n in enumerate(nodes):
        if ind == 0:
            node_ind_to_unfold = [2+ind, 2+ind+1] # unfold current and next node
        else:
            node_ind_to_unfold = [3*ind+5] # (+1 for initial partition, +1 transpose node, +3 partitions)

        model = model.transform(ExtendPartition(node_ind_to_unfold))
        model = model.transform(AbsorbTransposeIntoMultiThreshold())

        if ind==0:
            model = model.transform(PartitionFromDict(partitionings[0], build_dir+"/partitioning_repartition"))
        if ind==1:
            model = model.transform(PartitionFromDict(partitionings[1], build_dir+"/partitioning_repartition"))
        if ind==2:
            model = model.transform(PartitionFromDict(partitionings[2], build_dir+"/partitioning_repartition"))
        if ind==3:
            model = model.transform(PartitionFromDict(partitionings[3], build_dir+"/partitioning_repartition"))
        if ind==4:
            model = model.transform(PartitionFromDict(partitionings[4], build_dir+"/partitioning_repartition"))
            break

    model = model.transform(GiveUniqueNodeNames())
    model.save(build_dir+"/end2end_quartznet_lowered_partitioned.onnx")

def test_end2end_quartznet_convert_to_hls_layers():
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_lowered_partitioned.onnx")

    #partition_dir = build_dir+"/partitioning_hls"
    partition_id = 1
    nodes = [n for n in model.graph.node if (n.op_type=="GenericPartition" and n.name!="GenericPartition_0")]
    for n in nodes:
        #inst = GetCustomOp(n)
        prefix = "pt_"+str(partition_id)+"_"

        path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
        model_partition = ModelWrapper(path_to_partition)

        model_partition = model_partition.transform(to_hls.InferConvInpGen(), make_deepcopy=False)
        model_partition = model_partition.transform(to_hls.InferVVAU(), make_deepcopy=False)
        model_partition = model_partition.transform(to_hls.InferQuantizedStreamingFCLayer(mem_mode), make_deepcopy=False)
        model_partition = model_partition.transform(to_hls.InferThresholdingLayer(), make_deepcopy=False)
        model_partition = model_partition.transform(to_hls.InferAddStreamsLayer(), make_deepcopy=False)
        model_partition = model_partition.transform(to_hls.InferDuplicateStreamsLayer(), make_deepcopy=False)

        #pathlib.Path(self.partition_dir).mkdir(parents=True, exist_ok=True)
        #partition_path = partition_dir+"/partition_"+str(partition_id)+".onnx"
        model_partition.save(path_to_partition)
        #inst.set_nodeattr("model", partition_path)

        partition_id+=1

    model.save(build_dir+"/end2end_quartznet_hls_layers.onnx")


def test_end2end_create_dataflow_partition():
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_hls_layers.onnx")

    # Extend partitions, as the HLS model can now be contained in a single ONNX file
    list_of_partitions = list(range(0, 17))
    model = model.transform(ExtendPartition(list_of_partitions))
    # Give unique node names
    model = model.transform(GiveUniqueNodeNames())
    # Create fpgadataflow partition node for consistency
    model = model.transform(CreateDataflowPartition())

    model.save(build_dir+"/end2end_quartznet_dataflow_partition.onnx")


def test_end2end_quartznet_folding(filename="/end2end_quartznet_dataflow_partition.onnx", apply_config=True):
    model = load_test_checkpoint_or_skip(build_dir+filename)

    for n in model.graph.node:
        if n.op_type=="StreamingDataflowPartition":
            path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
            model_partition = ModelWrapper(path_to_partition)

            if apply_config:
                model_partition = model_partition.transform(ApplyConfig(build_dir+"/folding_config.json"))
            else:
                model_partition = model_partition.transform(SetFolding(target_cycles_per_frame=4200000))
                for n_par in model_partition.graph.node:
                    inst = getCustomOp(n_par)
                    if n_par.op_type in ["FMPadding_Batch", "DuplicateStreams_Batch", "AddStreams_Batch"]:
                        continue
                    elif n_par.op_type=="ConvolutionInputGenerator1D":
                        name = n_par.name
                        node_idx = int(name.split("_")[-1])
                        if node_idx>=31 and node_idx<=44: # node_idx 31-44
                            inst.set_nodeattr("ram_style", "block")
                        elif node_idx>=45 and node_idx<=59: # node_idx 45-59
                            inst.set_nodeattr("ram_style", "ultra")
                        else:
                            inst.set_nodeattr("ram_style", "distributed")
                    elif n_par.op_type=="Vector_Vector_Activate_Batch":
                        inst.set_nodeattr("resType", "dsp")
                        # Vivado HLS runs out of memory for VVAU nodes of 7th partition and onwards. Solution for now: increase PE to lower load/store instructions.
                        # TODO: find a better solution that does not require wasting resources.
                        name = n_par.name
                        node_idx = int(name.split("_")[-1])
                        ch = inst.get_nodeattr("Channels")
                        k = inst.get_nodeattr("Kernel")[0]
                        pe = inst.get_nodeattr("PE")
                        load_stores = ch*k/pe
                        # NOTE: preceding ConvInpGen must also be re-folded
                        prod = model_partition.find_producer(n_par.input[0])
                        inst_prod = getCustomOp(prod)
                        if node_idx>30 and node_idx<=44:
                            if load_stores > 13500:
                                new_pe = pe*2
                                if load_stores > 27000:
                                    new_pe = pe*4
                                inst.set_nodeattr("PE", new_pe)
                                inst_prod.set_nodeattr("SIMD", new_pe)
                                print("WARNING: load_stores exceeds 13500 ({}) for node {}, increasing PE from {} to {}!".format(load_stores, n_par.name, pe, new_pe))
                        if node_idx>44:
                            if load_stores > 8000:
                                new_pe = pe*2
                                if load_stores > 16000:
                                    new_pe = pe*4
                                if load_stores > 33000:
                                    new_pe = pe*8
                                inst.set_nodeattr("PE", new_pe)
                                inst_prod.set_nodeattr("SIMD", new_pe)
                                print("WARNING: load_stores exceeds 8000 ({}) for node {}, increasing PE from {} to {}!".format(load_stores, n_par.name, pe, new_pe))
                        else: # node_idx<=30
                            if load_stores > 32000:
                                new_pe = pe*2
                                inst.set_nodeattr("PE", new_pe)
                                inst_prod.set_nodeattr("SIMD", new_pe)
                                print("WARNING: load_stores exceeds 32000 ({}) for node {}, increasing PE from {} to {}!".format(load_stores, n_par.name, pe, new_pe))
                    elif n_par.op_type=="StreamingFCLayer_Batch":
                        inst.set_nodeattr("resType", "dsp")
                        inst.set_nodeattr("ram_style", "ultra")
                        inst.set_nodeattr("mem_mode", "decoupled")
                        if inst.get_nodeattr("ram_style")=="ultra":
                            inst.set_nodeattr("runtime_writeable_weights", 1)
                    elif n_par.op_type=="Thresholding_Batch":
                        inst.set_nodeattr("ram_style", "distributed")
                        inst.set_nodeattr("mem_mode", "const")
                    else:
                        print("Missed: {} in folding!".format(n_par.op_type))
                        break

            model_partition.save(path_to_partition)

    model.save(build_dir+"/end2end_quartznet_folded.onnx")


def test_end2end_quartznet_cppsim(verify=False):
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_folded.onnx")

    start = time.time()
    for n in model.graph.node:
        if n.op_type=="StreamingDataflowPartition":
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
    if verify:
        verify_step(model, "cppsim")


def test_end2end_quartznet_generate_estimate_reports():
    # See https://github.com/Xilinx/finn/blob/dev/src/finn/builder/build_dataflow_steps.py
    #model = load_test_checkpoint_or_skip(build_dir + "/end2end_quartznet_cppsim.onnx")
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_quartznet_folded.onnx")
    report_dir = build_dir + "/report"
    os.makedirs(report_dir, exist_ok=True)

    for n in model.graph.node:
        if n.op_type=="StreamingDataflowPartition":
            path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
            model_partition = ModelWrapper(path_to_partition)
            ops_and_params = model_partition.analysis(op_and_param_counts)
            with open(report_dir + "/op_and_param_counts.json", "w") as f:
                json.dump(ops_and_params, f, indent=2)
            estimate_layer_cycles = model_partition.analysis(exp_cycles_per_layer)
            with open(report_dir + "/estimate_layer_cycles.json", "w") as f:
                json.dump(estimate_layer_cycles, f, indent=2)
            estimate_layer_resources = model_partition.analysis(res_estimation)
            estimate_layer_resources["total"] = aggregate_dict_keys(
                estimate_layer_resources
            )
            with open(report_dir + "/estimate_layer_resources.json", "w") as f:
                json.dump(estimate_layer_resources, f, indent=2)
            estimate_layer_resources_complete = model_partition.analysis(res_estimation_complete)
            with open(report_dir + "/estimate_layer_config_alternatives.json", "w") as f:
                json.dump(estimate_layer_resources_complete, f, indent=2)
            # need to call AnnotateCycles before dataflow_performance
            model_partition = model_partition.transform(AnnotateCycles())
            estimate_network_performance = model_partition.analysis(dataflow_performance)
            # add some more metrics to estimated performance
            n_clock_cycles_per_sec = (10 ** 9) / target_clk_ns
            est_fps = n_clock_cycles_per_sec / estimate_network_performance["max_cycles"]
            estimate_network_performance["estimated_throughput_fps"] = est_fps
            est_latency_ns = (
                estimate_network_performance["critical_path_cycles"]
                * target_clk_ns
            )
            estimate_network_performance["estimated_latency_ns"] = est_latency_ns
            with open(report_dir + "/estimate_network_performance.json", "w") as f:
                json.dump(estimate_network_performance, f, indent=2)

            model_partition.save(path_to_partition)


def test_end2end_quartznet_gen_hls_ip():
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_folded.onnx")

    start = time.time()
    for n in model.graph.node:
        if n.op_type=="StreamingDataflowPartition":
            path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
            model_partition = ModelWrapper(path_to_partition)

            model_partition = model_partition.transform(PrepareIP(test_fpga_part, target_clk_ns))
            model_partition.save(path_to_partition)
            model_partition = model_partition.transform(HLSSynthIP())
            model_partition.save(path_to_partition)
            model_partition = model_partition.transform(ReplaceVerilogRelPaths())
            model_partition = model_partition.transform(AnnotateResources("hls"))
            model_partition.save(path_to_partition)

            report_dir = build_dir + "/report"
            os.makedirs(report_dir, exist_ok=True)
            estimate_layer_resources_hls = model_partition.analysis(hls_synth_res_estimation)
            with open(report_dir + "/estimate_layer_resources_hls.json", "w") as f:
                json.dump(estimate_layer_resources_hls, f, indent=2)

    end = time.time()

    elapsed_time = end - start
    f = open(build_dir + "/end2end_quartznet_ipgen_time.txt", "w+")
    f.write("Execution time in seconds: " + str(elapsed_time))
    f.close()

    model.save(build_dir+"/end2end_quartznet_ipgen.onnx")


def test_end2end_quartznet_set_fifo_depths(manually=True):
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_ipgen.onnx")

    start = time.time()
    for n in model.graph.node:
        if n.op_type=="StreamingDataflowPartition":
            inst = getCustomOp(n)
            path_to_partition = inst.get_nodeattr("model")
            new_file = path_to_partition.replace("df_model", "df_model_fifos")
            print("New file will be located at: {} -> {}".format(path_to_partition, new_file))
            model_partition = ModelWrapper(path_to_partition)

            if manually:
                model_partition = model_partition.transform(InsertDWC())
                #model_partition = model_partition.transform(InsertFIFO(create_shallow_fifos=True))
                model_partition = model_partition.transform(GiveUniqueNodeNames())
                model_partition = model_partition.transform(GiveReadableTensorNames())
                #for n_par in model_partition.graph.node:
                #    if n_par.op_type=="StreamingFIFO":
                #        inst_par = getCustomOp(n_par)
                #        inst_par.set_nodeattr("depth", 256)
                #        inst_par.set_nodeattr("impl_style", "vivado")
                #        inst_par.set_nodeattr("ram_style", "auto")
                #model_partition = model_partition.transform(RemoveShallowFIFOs())
                model_partition.save(new_file)
            else:
                # TODO: add Vivado ram style after inspecting resource utilization --> probably LUTs (from back-on-the-envelope resource estimation calculations)
                # max_qsrl_depth=1 to always ensure Vivado IP is used (impl_style=Vivado)
                model_partition = model_partition.transform(InsertAndSetFIFODepths(test_fpga_part, target_clk_ns, max_qsrl_depth=1, vivado_ram_style="auto"))
                model_partition.save(new_file)

            end = time.time()
            elapsed_time = end-start
            f = open(build_dir + "/end2end_quartznet_setfifo_time.txt", "w+")
            f.write("Execution time in seconds: " + str(elapsed_time))
            f.close()

            # extract the final configuration and save it as json
            hw_attrs = [
                "PE",
                "SIMD",
                "ram_style",
                "depth",
                "impl_style",
                "resType",
                "mem_mode",
                "runtime_writeable_weights",
            ]

            print("Calling PrepareIP and HLSSynthIP again...")
            report_dir = build_dir + "/report"
            os.makedirs(report_dir, exist_ok=True)
            extract_model_config_to_json(
                model_partition, report_dir + "/final_hw_config.json", hw_attrs
            )

            ## after FIFOs are ready to go, call PrepareIP and HLSSynthIP again
            model_partition = model_partition.transform(PrepareIP(test_fpga_part, target_clk_ns))
            model_partition.save(new_file)
            model_partition = model_partition.transform(HLSSynthIP())
            model_partition.save(new_file)
            # Change model attribute of StreamingDataflowPartition
            inst.set_nodeattr("model", new_file)

    model.save(build_dir+"/end2end_quartznet_fifos.onnx")


def test_end2end_quartznet_create_stitched_ip():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_quartznet_fifos.onnx")

    start = time.time()
    for n in model.graph.node:
        if n.op_type=="StreamingDataflowPartition":
            inst = getCustomOp(n)
            path_to_partition = inst.get_nodeattr("model")
            new_file = path_to_partition.replace("df_model_fifos", "df_model_stitchedip")
            print("New file will be located at: {} -> {}".format(path_to_partition, new_file))
            model_partition = ModelWrapper(path_to_partition)
            model_partition = model_partition.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
            model_partition.save(new_file)
            inst.set_nodeattr("model", new_file)

    end = time.time()
    elapsed_time = end-start
    f = open(build_dir + "/end2end_quartznet_stitchedip_time.txt", "w+")
    f.write("Execution time in seconds: " + str(elapsed_time))
    f.close()

    model.save(build_dir + "/end2end_quartznet_stitched_ip.onnx")


def test_end2end_quartznet_rtlsim():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_quartznet_stitched_ip.onnx")

    for n in model.graph.node:
        if n.op_type=="StreamingDataflowPartition":
            inst = getCustomOp(n)
            path_to_partition = inst.get_nodeattr("model")
            new_file = path_to_partition.replace("df_model_stitchedip", "df_model_rtlsim")
            model_partition = ModelWrapper(path_to_partition)
            model_partition = model_partition.transform(AnnotateCycles())
            perf = model_partition.analysis(dataflow_performance)
            latency = perf["critical_path_cycles"]
            os.environ["LIVENESS_THRESHOLD"] = str(int(latency * 5))

            verify_model = deepcopy(model_partition)
            # rtlsim only supports impl_style=rtl for StreamingFIFO, ensure that
            for fifo_layer in verify_model.get_nodes_by_op_type("StreamingFIFO"):
                getCustomOp(fifo_layer).set_nodeattr("impl_style", "rtl")
                getCustomOp(fifo_layer).set_nodeattr("code_gen_dir_ipgen", "")
                getCustomOp(fifo_layer).set_nodeattr("ipgen_path", "")
                getCustomOp(fifo_layer).set_nodeattr("ip_path", "")
                getCustomOp(fifo_layer).set_nodeattr("ip_vlnv", "")
            # similarly for StreamingDataWidthConverter with impl_style=hls
            for dwc_layer in verify_model.get_nodes_by_op_type("StreamingDataWidthConverter_Batch"):
                getCustomOp(dwc_layer).set_nodeattr("impl_style", "hls")
                getCustomOp(dwc_layer).set_nodeattr("code_gen_dir_ipgen", "")
                getCustomOp(dwc_layer).set_nodeattr("ipgen_path", "")
                getCustomOp(dwc_layer).set_nodeattr("ip_path", "")
                getCustomOp(dwc_layer).set_nodeattr("ip_vlnv", "")
            verify_model = verify_model.transform(PrepareIP(test_fpga_part, target_clk_ns))
            verify_model = verify_model.transform(HLSSynthIP())
            verify_model = verify_model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
            verify_model = verify_model.transform(PrepareRTLSim())
            verify_model.set_metadata_prop("exec_mode", "rtlsim")
            verify_model.save(new_file)
            inst.set_nodeattr("model", new_file)

    model.save(build_dir + "/end2end_quartznet_rtlsim.onnx")
    start = time.time()
    verify_step(model, "stitched_ip_rtlsim")
    end = time.time()
    elapsed_time = end-start
    f = open(build_dir + "/end2end_quartznet_rtlsim_time.txt", "w+")
    f.write("Execution time in seconds: " + str(elapsed_time))
    f.close()


def test_end2end_quartznet_rtlsim_performance():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_quartznet_stitched_ip.onnx")

    for n in model.graph.node:
        if n.op_type=="StreamingDataflowPartition":
            path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
            model_partition = ModelWrapper(path_to_partition)
            # prepare ip-stitched rtlsim
            rtlsim_model = deepcopy(model_partition)
            # rtlsim only supports impl_style=rtl for StreamingFIFO, ensure that
            for fifo_layer in rtlsim_model.get_nodes_by_op_type("StreamingFIFO"):
                getCustomOp(fifo_layer).set_nodeattr("impl_style", "rtl")
            # similarly for StreamingDataWidthConverter with impl_style=hls
            for dwc_layer in rtlsim_model.get_nodes_by_op_type("StreamingDataWidthConverter_Batch"):
                getCustomOp(dwc_layer).set_nodeattr("impl_style", "hls")
            rtlsim_model = rtlsim_model.transform(PrepareRTLSim())
            rtlsim_model.set_metadata_prop("exec_mode", "rtlsim")
            # run with single input to get latency
            rtlsim_perf_dict = throughput_test_rtlsim(rtlsim_model, 1)
            rtlsim_latency = rtlsim_perf_dict["cycles"]
            # run with num inputs equal to layers to fill the whole pipeline
            # to get the steady-state throughput
            rtlsim_bs = len(rtlsim_model.graph.node)
            rtlsim_perf_dict = throughput_test_rtlsim(rtlsim_model, rtlsim_bs)
            rtlsim_perf_dict["latency_cycles"] = rtlsim_latency
            report_dir = build_dir + "/report"
            os.makedirs(report_dir, exist_ok=True)
            with open(report_dir + "/rtlsim_performance.json", "w") as f:
                json.dump(rtlsim_perf_dict, f, indent=2)


def test_end2end_quartznet_ooc(verify=False):
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_quartznet_stitched_ip.onnx")

    start = time.time()
    for n in model.graph.node:
        if n.op_type=="StreamingDataflowPartition":
            inst = getCustomOp(n)
            path_to_partition = inst.get_nodeattr("model")
            new_file = path_to_partition.replace("df_model_stitchedip", "df_model_ooc")
            print("New file will be located at: {} -> {}".format(path_to_partition, new_file))
            model_partition = ModelWrapper(path_to_partition)
            model_partition = model_partition.transform(SynthOutOfContext(test_fpga_part, target_clk_ns))
            model_partition.save(new_file)

            end = time.time()
            elapsed_time = end-start
            f = open(build_dir + "/end2end_quartznet_ooc_synthesis_time.txt", "w+")
            f.write("Execution time in seconds: " + str(elapsed_time))
            f.close()
            # Save model
            inst.set_nodeattr("model", new_file)
            model.save(build_dir + "/end2end_quartznet_ooc.onnx")
            # Log resource and estimated throughput
            report_dir = build_dir + "/report"
            os.makedirs(report_dir, exist_ok=True)
            ooc_res_dict = model_partition.get_metadata_prop("res_total_ooc_synth")
            ooc_res_dict = eval(ooc_res_dict)
            estimate_network_performance = model_partition.analysis(dataflow_performance)
            # add some more metrics to estimated performance
            n_clock_cycles_per_sec = float(ooc_res_dict["fmax_mhz"]) * (10 ** 6)
            est_fps = n_clock_cycles_per_sec / estimate_network_performance["max_cycles"]
            ooc_res_dict["estimated_throughput_fps"] = est_fps
            with open(report_dir + "/ooc_synth_and_timing.json", "w") as f:
                json.dump(ooc_res_dict, f, indent=2)

    model.save(build_dir + "/end2end_quartznet_ooc.onnx")


def test_end2end_quartznet_create_bitfile(slr_file):
    #model = load_test_checkpoint_or_skip(build_dir + "/end2end_quartznet_stitched_ip.onnx")
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_quartznet_fifos.onnx")

    bitfile_dir = build_dir + "/bitfile"
    os.makedirs(bitfile_dir, exist_ok=True)
    report_dir = build_dir + "/report"
    os.makedirs(report_dir, exist_ok=True)
    start = time.time()
    for n in model.graph.node:
        if n.op_type=="StreamingDataflowPartition":
            inst = getCustomOp(n)
            path_to_partition = inst.get_nodeattr("model")
            #new_file = path_to_partition.replace("df_model_stitchedip", "df_model_bitfile")
            new_file = path_to_partition.replace("df_model_fifos", "df_model_bitfile")
            print("New file will be located at: {} -> {}".format(path_to_partition, new_file))
            model_partition = ModelWrapper(path_to_partition)
            model_partition = model_partition.transform(ApplyConfig(slr_file)) # apply SLR assignment
            model_partition = model_partition.transform(VitisBuild(
                test_fpga_part, target_clk_ns, test_platform, VitisOptStrategy.PERFORMANCE_BEST, enable_debug=False, enable_link=True, parent_model_path=new_file)
                )
            model_partition.save(new_file)
            #copy(model_partition.get_metadata_prop("bitfile"), bitfile_dir + "/finn-accel.xclbin")
            #copy(model_partition.get_metadata_prop("vivado_synth_rpt"), report_dir + "/post_synth_resources.xml")
    end = time.time()
    elapsed_time = end-start
    f = open(build_dir + "/end2end_quartznet_bitfile_generation_time.txt", "w+")
    f.write("Execution time in seconds: " + str(elapsed_time))
    f.close()

    model.save(build_dir + "/end2end_quartznet_bitfile.onnx")


def test_vitisbuild():
    print("Creating bitfile")
    slr_file = build_dir+"/manual_slr_partitioning.json"
    test_end2end_quartznet_create_bitfile(slr_file)

def test_cppsim():
    print("CPPsim")
    test_end2end_quartznet_cppsim(verify=True)
    print("Generate estimate reports")
    test_end2end_quartznet_generate_estimate_reports()

def test_refold_cppsim():
    # Set filename to either "/end2end_quartznet_dataflow_partition.onnx" or "/end2end_quartznet_ipgen.onnx" (refolding)
    print("Folding")
    filename = "/end2end_quartznet_ipgen.onnx"
    apply_config = True
    test_end2end_quartznet_folding(filename, apply_config)
    test_cppsim()

def test_ooc():
    print("Create stitched IP")
    test_end2end_quartznet_create_stitched_ip()
    print("Out-of-context synthesis")
    test_end2end_quartznet_ooc()

def test_refold_ooc():
    # Set filename to either "/end2end_quartznet_dataflow_partition.onnx" or "/end2end_quartznet_ipgen.onnx" (refolding)
    print("Folding")
    filename = "/end2end_quartznet_ipgen.onnx"
    apply_config = True
    test_end2end_quartznet_folding(filename, apply_config)
    print("Generate RTL and HLS synthesis")
    test_end2end_quartznet_gen_hls_ip()
    ## NOTE: set manually parameter (i.e. disable InsertAndSetFIFODepths)
    manually=True
    print("Set FIFO depths")
    test_end2end_quartznet_set_fifo_depths(manually)
    test_ooc()

def test_rtlsim():
    print("RTLsim")
    test_end2end_quartznet_rtlsim()
    #print("RTLsim performance")
    #test_end2end_quartznet_rtlsim_performance()


def test_all():
    print("Brevitas export")
    test_end2end_quartznet_brevitas_export()
    print("Tidy and change shape tensors")
    test_end2end_quartznet_tidy_and_change_shape_tensors(verify=True)
    print("Streamline")
    test_end2end_quartznet_streamline(verify=True)
    print("Lowering")
    test_end2end_quartznet_lowering(verify=True)
    print("Repartition")
    test_end2end_quartznet_repartition(verify=True)
    print("Convert to HLS layers")
    test_end2end_quartznet_convert_to_hls_layers()
    print("Create dataflow partition")
    test_end2end_create_dataflow_partition()

    # Set filename to either "/end2end_quartznet_dataflow_partition.onnx" or "/end2end_quartznet_ipgen.onnx" (refolding)
    print("Folding")
    filename = "/end2end_quartznet_dataflow_partition.onnx"
    apply_config = True
    test_end2end_quartznet_folding(filename, apply_config)

    #print("CPPsim")
    #test_end2end_quartznet_cppsim(verify=True)
    #print("Generate estimate reports")
    #test_end2end_quartznet_generate_estimate_reports()

    print("Generate RTL and HLS synthesis")
    test_end2end_quartznet_gen_hls_ip()

    ## NOTE: set manually parameter (i.e. disable InsertAndSetFIFODepths)
    manually=True
    print("Set FIFO depths")
    test_end2end_quartznet_set_fifo_depths(manually)

    #print("Create stitched IP")
    #test_end2end_quartznet_create_stitched_ip()

    #print("RTLsim")
    #test_end2end_quartznet_rtlsim()
    #print("RTLsim performance")
    #test_end2end_quartznet_rtlsim_performance()

    #print("Out-of-context synthesis")
    #test_end2end_quartznet_ooc()

    #print("Creating bitfile")
    #slr_file = build_dir+"/manual_slr_partitioning.json"
    #test_end2end_quartznet_create_bitfile(slr_file)
