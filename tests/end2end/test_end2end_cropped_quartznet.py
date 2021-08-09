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
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
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

build_dir = os.environ["FINN_BUILD_DIR"]
mem_mode = "decoupled"
test_board = "U250"
test_platform = alveo_default_platform[test_board]
test_fpga_part = alveo_part_map[test_board]
target_clk_ns = 5


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
        output_dict = oxe.execute_onnx(model, input_dict)
        produced = output_dict[oname]
        produced = np.reshape(produced, np.shape(produced)+(1,))
        # Save QuartzNet export output, which will be used as reference for other transformations
        np.save(build_dir+"/librispeech_data/quartznet_export_output.npy", produced)
        ## Execute Pytorch/Brevitas simulation
        #inp_val_torch = torch.from_numpy(inp_val).float()
        ## Do forward pass and compare output
        expected = np.load(build_dir+"/librispeech_data/output_sample_float_0.npy")
        #expected = quartznet_torch.forward(inp_val_torch).detach().numpy()
    else:
        iname = model.graph.input[0].name
        oname = model.graph.output[0].name
        inp_val = np.load(build_dir+"/librispeech_data/input_sample_0.npy")
        inp_val = np.reshape(inp_val, np.shape(inp_val)+(1,)) # make 4D tensor
        input_dict = {iname: inp_val}
        # Execute FINN simulation
        output_dict = oxe.execute_onnx(model, input_dict)
        produced = output_dict[oname]
        np.save(build_dir+"/librispeech_data/quartznet_"+step_name+".npy", produced)
        # Compare against golden output (QuartzNet exported model)
        #expected = np.load(build_dir+"/librispeech_data/quartznet_export_output.npy")

    #res = np.isclose(expected, produced, atol=1e-3).all() # basically the same as np.array_equal(expected, produced) for integer outputs
    #res_to_str = {True: "SUCCESS", False: "FAIL"}
    #res_str = res_to_str[res]
    res_str = "DONE"
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
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_cropped_quartznet_hls_layers.onnx")

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(CreateDataflowPartition())

    model.save(build_dir+"/end2end_quartznet_dataflow_partition.onnx")


def test_end2end_quartznet_folding(verify=False):
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_dataflow_partition.onnx")

    for n in model.graph.node:
        if n.op_type=="StreamingDataflowPartition":
            path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
            model_partition = ModelWrapper(path_to_partition)
            model_partition = model_partition.transform(SetFolding(target_cycles_per_frame=4200000))

            for n_par in model_partition.graph.node:
                inst = getCustomOp(n_par)
                if n_par.op_type in ["FMPadding_Batch", "DuplicateStreams_Batch", "AddStreams_Batch"]:
                    continue
                elif n_par.op_type=="ConvolutionInputGenerator1D":
                    inst.set_nodeattr("ram_style", "distributed")
                elif n_par.op_type=="Vector_Vector_Activate_Batch":
                    inst.set_nodeattr("resType", "dsp")
                    # Vivado HLS runs out of memory for VVAU nodes of 7th partition and onwards. Solution for now: increase PE to lower load/store instructions.
                    # TODO: find a better solution that does not require wasting resources.
                    if n_par.name=="Vector_Vector_Activate_Batch_8":
                        inst.set_nodeattr("PE", 8)
                    if n_par.name=="Vector_Vector_Activate_Batch_5":
                        inst.set_nodeattr("PE", 2)
                    if n_par.name=="Vector_Vector_Activate_Batch_6":
                        inst.set_nodeattr("PE", 4)
                    if n_par.name=="Vector_Vector_Activate_Batch_7":
                        inst.set_nodeattr("PE", 8)
                    #name = n_par.name
                    #node_idx = int(name.split("_")[-1])
                    #ch = inst.get_nodeattr("Channels")
                    #k = inst.get_nodeattr("Kernel")[0]
                    #pe = inst.get_nodeattr("PE")
                    #load_stores = ch*k/pe
                    #if node_idx>30 and node_idx<=44:
                    #    if load_stores > 13500:
                    #        new_pe = pe*2
                    #        if load_stores > 27000:
                    #            new_pe = pe*4
                    #        inst.set_nodeattr("PE", new_pe)
                    #        print("WARNING: load_stores exceeds 13500 ({}) for node {}, increasing PE from {} to {}!".format(load_stores, n_par.name, pe, new_pe))
                    #if node_idx>44:
                    #    if load_stores > 8000:
                    #        new_pe = pe*2
                    #        if load_stores > 16000:
                    #            new_pe = pe*4
                    #        if load_stores > 33000:
                    #            new_pe = pe*8
                    #        inst.set_nodeattr("PE", new_pe)
                    #        print("WARNING: load_stores exceeds 8000 ({}) for node {}, increasing PE from {} to {}!".format(load_stores, n_par.name, pe, new_pe))
                    #else: # node_idx<=30
                    #    if load_stores > 32000:
                    #        new_pe = pe*2
                    #        inst.set_nodeattr("PE", new_pe)
                    #        print("WARNING: load_stores exceeds 16000 ({}) for node {}, increasing PE from {} to {}!".format(load_stores, n_par.name, pe, new_pe))
                elif n_par.op_type=="StreamingFCLayer_Batch":
                    inst.set_nodeattr("resType", "dsp")
                    inst.set_nodeattr("ram_style", "ultra")
                    #inst.set_nodeattr("ram_style", "block")
                    inst.set_nodeattr("mem_mode", "decoupled")
                    #inst.set_nodeattr("mem_mode", "const")
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
    #model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_cppsim.onnx")
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_folded.onnx")

    start = time.time()
    for n in model.graph.node:
        if n.op_type=="StreamingDataflowPartition":
            path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
            model_partition = ModelWrapper(path_to_partition)

            model_partition = model_partition.transform(PrepareIP(test_fpga_part, target_clk_ns))
            model_partition.save(path_to_partition)
            break
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


def test_end2end_quartznet_set_fifo_depths():
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_ipgen.onnx")

    start = time.time()
    for n in model.graph.node:
        if n.op_type=="StreamingDataflowPartition":
            path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
            model_partition = ModelWrapper(path_to_partition)
            # TODO: add Vivado ram style after inspecting resource utilization --> probably LUTs (from back-on-the-envelope resource estimation calculations)
            model_partition = model_partition.transform(InsertAndSetFIFODepths(test_fpga_part, target_clk_ns))
            model_partition.save(path_to_partition)

    end = time.time()
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

    for n in model.graph.node:
        if n.op_type=="StreamingDataflowPartition":
            path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
            model_partition = ModelWrapper(path_to_partition)
            extract_model_config_to_json(
                model_partition, build_dir + "/final_hw_config.json", hw_attrs
            )

            start = time.time()
            # after FIFOs are ready to go, call PrepareIP and HLSSynthIP again
            model_partition = model_partition.transform(PrepareIP(test_fpga_part, target_clk_ns))
            model_partition.save(path_to_partition)
            model_partition = model_partition.transform(HLSSynthIP())
            model_partition.save(path_to_partition)
            end = time.time()
            f = open(build_dir + "/end2end_quartznet_fifo_hlssynth.txt", "w+")
            f.write("Execution time in seconds: " + str(elapsed_time))
            f.close()

    model.save(build_dir+"/end2end_quartznet_fifos.onnx")


def test_end2end_quartznet_create_stitched_ip():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_quartznet_fifos.onnx")

    start = time.time()
    for n in model.graph.node:
        if n.op_type=="StreamingDataflowPartition":
            path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
            model_partition = ModelWrapper(path_to_partition)
            model_partition = model_partition.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
            model_partition.save(path_to_partition)

    end = time.time()
    f = open(build_dir + "/end2end_quartznet_stitchedip_time.txt", "w+")
    f.write("Execution time in seconds: " + str(elapsed_time))
    f.close()

    model.save(build_dir + "/end2end_quartznet_stitched_ip.onnx")


def test_end2end_quartznet_rtlsim():
    model = load_test_checkpoint_or_skip(build_dir + "/end2end_quartznet_stitched_ip.onnx")

    start = time.time()
    for n in model.graph.node:
        if n.op_type=="StreamingDataflowPartition":
            path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
            model_partition = ModelWrapper(path_to_partition)
            verify_model = deepcopy(model_partition)
            # rtlsim only supports impl_style=rtl for StreamingFIFO, ensure that
            for fifo_layer in verify_model.get_nodes_by_op_type("StreamingFIFO"):
                getCustomOp(fifo_layer).set_nodeattr("impl_style", "rtl")
            # similarly for StreamingDataWidthConverter with impl_style=hls
            for dwc_layer in verify_model.get_nodes_by_op_type("StreamingDataWidthConverter_Batch"):
                getCustomOp(dwc_layer).set_nodeattr("impl_style", "hls")
            verify_model = verify_model.transform(PrepareRTLSim())
            verify_model.set_metadata_prop("exec_mode", "rtlsim")
            verify_step(verify_model, "stitched_ip_rtlsim")
    end = time.time()
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
            path_to_partition = get_by_name(n.attribute, "model", "name").s.decode('utf-8')
            model_partition = ModelWrapper(path_to_partition)
            model_partition = model_partition.transform(SynthOutOfContext(test_fpga_part, target_clk_ns))
            model_partition.save(path_to_partition)

            end = time.time()
            f = open(build_dir + "/end2end_quartznet_ooc_synthesis_time.txt", "w+")
            f.write("Execution time in seconds: " + str(elapsed_time))
            f.close()

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


def test_all():
    #print("Brevitas export")
    #test_end2end_quartznet_brevitas_export()
    #print("Tidy and change shape tensors")
    #test_end2end_quartznet_tidy_and_change_shape_tensors(verify=True)
    #print("Streamline")
    #test_end2end_quartznet_streamline(verify=True)
    #print("Lowering")
    #test_end2end_quartznet_lowering(verify=True)
    #print("Repartition")
    #test_end2end_quartznet_repartition(verify=True)
    #print("Convert to HLS layers")
    #test_end2end_quartznet_convert_to_hls_layers()

    print("Create dataflow partition")
    test_end2end_create_dataflow_partition()
    print("Folding")
    test_end2end_quartznet_folding()
    #print("CPPsim")
    #test_end2end_quartznet_cppsim(verify=True)
    print("Generate estimate reports")
    test_end2end_quartznet_generate_estimate_reports()
    print("Generate RTL and HLS synthesis")
    test_end2end_quartznet_gen_hls_ip()

    #print("Set FIFO depths")
    #test_end2end_quartznet_set_fifo_depths()
    #print("Create stitched IP")
    #test_end2end_quartznet_create_stitched_ip()
    #print("RTLsim")
    #test_end2end_quartznet_rtlsim()
    #print("RTLsim performance"
    #test_end2end_quartznet_rtlsim_performance()
    #print("Out-of-context synthesis")
    #test_end2end_quartznet_ooc()
