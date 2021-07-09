# Copyright (c) 2020, Xilinx
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

import copy

from onnx import helper
from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation
from finn.util.basic import get_by_name, make_build_dir


class CreateDataflowPartition(Transformation):
    """Split a graph into two graphs; one which contains non-FINN-dataflow nodes
    and a StreamingDataflowPartition node, and another which only contains
    FINN dataflow nodes. The StreamingDataflowPartition has a model attribute
    that indicates the filename for the second graph that only contains
    dataflow nodes. No action is taken if there are no dataflow nodes."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        target_partition_id = 0
        # we currently assume that all dataflow nodes belonging to the same partition
        # are connected to each other and there is a single input/output to/from each.
        # NOTE: all dataflow nodes with no partition_id set are moved to partition 0
        # TODO: check the assumption and/or improve this.
        while True:
            all_nodes = list(model.graph.node)
            df_nodes = filter(
                lambda x: get_by_name(x.attribute, "backend") is not None, all_nodes
            )
            df_nodes = filter(
                lambda x: get_by_name(x.attribute, "backend").s.decode("UTF-8")
                == "fpgadataflow"
                and (
                    get_by_name(x.attribute, "partition_id") is None
                    or get_by_name(x.attribute, "partition_id").i == target_partition_id
                )
                and x.op_type != "StreamingDataflowPartition",
                df_nodes,
            )
            df_nodes = list(df_nodes)
            non_df_nodes = filter(lambda x: x not in df_nodes, all_nodes)
            non_df_nodes = list(non_df_nodes)

            if len(df_nodes) == 0:
                # no changes if no dataflow nodes are present
                break
            else:
                # partition the model into two models
                df_model = copy.deepcopy(model)
                non_df_model = model
                # remove all non-dataflow nodes from the dataflow model
                for node_to_remove in non_df_nodes:
                    df_model.graph.node.remove(node_to_remove)
                # identify the entry and exit points for the dataflow part
                df_in = []
                df_out = []
                for node in df_model.graph.node:
                    for in_tensor in node.input:
                        # check if producer has been removed = lies outside the partition
                        has_initializer = in_tensor in [
                            x.name for x in df_model.graph.initializer
                        ]
                        has_producer = df_model.find_producer(in_tensor) is not None
                        if not has_initializer and not has_producer:
                            # the same tensor could feed multiple nodes within the partition
                            # (e.g. for residual connections), so we avoid duplicates
                            if in_tensor not in df_in:
                                df_in.append(in_tensor)
                    for out_tensor in node.output:
                        # check if tensor is top-level output
                        # or has a consumer outside the partition
                        if out_tensor in [x.name for x in model.graph.output]:
                            if out_tensor not in df_out:
                                df_out.append(out_tensor)
                        else:
                            for consumer in model.find_consumers(out_tensor):
                                if consumer in non_df_nodes and out_tensor not in df_out:
                                    df_out.append(out_tensor)

                df_in_vi = list(map(lambda x: df_model.get_tensor_valueinfo(x), df_in))
                df_out_vi = list(map(lambda x: df_model.get_tensor_valueinfo(x), df_out))

                for x in df_model.graph.input:
                    df_model.graph.input.remove(x)
                for i in df_in_vi:
                    df_model.graph.input.append(i)

                for x in df_model.graph.output:
                    df_model.graph.output.remove(x)
                for o in df_out_vi:
                    df_model.graph.output.append(o)

                # remove redundant input and output value_info entries
                for i in df_in_vi:
                    # the tensor can be both an input and value_info, so we also have to
                    # ensure that the tensor is not a relevant value_info before removing
                    if (
                        i in df_model.graph.value_info
                        and df_model.find_producer(i.name) is None
                    ):
                        df_model.graph.value_info.remove(i)

                for o in df_out_vi:
                    # the tensor can both an output and value_info, so we also have to
                    # ensure that the tensor is not a relevant value_info before removing
                    if (
                        o in df_model.graph.value_info
                        and df_model.find_consumers(o.name) is None
                    ):
                        df_model.graph.value_info.remove(o)

                #df_in = df_model.graph.node[0].input[0]
                #df_out = df_model.graph.node[-1].output[0]
                #df_in_vi = df_model.get_tensor_valueinfo(df_in)
                #df_out_vi = df_model.get_tensor_valueinfo(df_out)
                # set df graph in/out to be df_in/df_out
                #df_model.graph.input.remove(df_model.graph.input[0])
                #df_model.graph.input.insert(0, df_in_vi)
                #df_model.graph.output.remove(df_model.graph.output[0])
                #df_model.graph.output.insert(0, df_out_vi)

                # parse StreamingFCLayers looking for external weight memories
                fc_extw_nodes = filter(
                    lambda x: x.op_type == "StreamingFCLayer_Batch"
                    and get_by_name(x.attribute, "mem_mode") is not None
                    and get_by_name(x.attribute, "mem_mode").s.decode("UTF-8")
                    == "external",
                    df_nodes,
                )
                fc_extw_nodes = list(fc_extw_nodes)
                extra_df_inputs = []

                num_of_global_inputs = len(df_model.graph.input)
                for i in range(len(fc_extw_nodes)):
                    fc_weight_vi = df_model.get_tensor_valueinfo(
                        fc_extw_nodes[i].input[1]
                    )
                    df_model.graph.input.insert(i + num_of_global_inputs, fc_weight_vi)
                    extra_df_inputs.append(fc_extw_nodes[i].input[1])

                # save model
                df_model_dir = make_build_dir(
                    "dataflow_partition" + str(target_partition_id) + "_"
                )
                df_model_filename = df_model_dir + "/df_model.onnx"
                df_model.cleanup()
                df_model.save(df_model_filename)
                # remove all dataflow nodes from the non-dataflow model
                # keep track of where the dataflow part starts
                df_start_ind = all_nodes.index(df_nodes[0])

                # get and check floorplan
                inst = getCustomOp(df_nodes[0])
                slr = inst.get_nodeattr("slr")
                for node in df_nodes[1:]:
                    inst = getCustomOp(node)
                    print("{}\t{}".format(node.name, inst.get_nodeattr("slr")))
                    assert slr == inst.get_nodeattr(
                        "slr"
                    ), """all nodes with
                same partition_id must have the same slr id"""
                print("{}-------".format(target_partition_id))

                # check that there is only one non-null mem_port per partition
                nmemports = 0
                mem_port = ""
                for node in df_nodes:
                    inst = getCustomOp(node)
                    port = inst.get_nodeattr("mem_port")
                    if port is not None and port != "":
                        nmemports += 1
                        mem_port = port
                assert nmemports <= 1, """too many memory ports per partition"""

                for node_to_remove in df_nodes:
                    non_df_model.graph.node.remove(node_to_remove)
                # create StreamingDataflow node with df_in/df_out io
                df_node = helper.make_node(
                    "StreamingDataflowPartition",
                    df_in + extra_df_inputs,
                    df_out,
                    # use the model attribute to mark the df model
                    model=df_model_filename,
                    domain="finn.custom_op.general",
                    partition_id=target_partition_id,
                    slr=slr,
                    mem_port=mem_port,
                )
                non_df_model.graph.node.insert(df_start_ind, df_node)
                model = non_df_model

                #if target_partition_id==4:
                #    return (model, False)
                target_partition_id += 1

        return (model, False)
