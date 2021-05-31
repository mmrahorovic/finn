import numpy as np
from finn.core.modelwrapper import ModelWrapper
from finn.util.basic import gen_finn_dt_tensor
import finn.core.onnx_exec as oxe
import time
import os
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode

build_dir = os.environ["FINN_BUILD_DIR"]


def exec_base_model():
    t1 = time.perf_counter()
    model_1 = ModelWrapper(build_dir+"/end2end_quartznet_export_dev.onnx")
    # Create input data
    input0_tensor_name = model_1.graph.input[0].name
    ## Change input
    input_val = np.load(build_dir+"/brevitas_reference/end2end_quartznet_input_quantized.npy")
    #input_val = input_val[:,:,0:256]
    input_dict = {}
    input_dict[input0_tensor_name] = input_val
    output0_tensor_name = model_1.graph.output[0].name
    expected_m1_dict = oxe.execute_onnx(model_1, input_dict, return_full_exec_context = False)
    expected_m1 = expected_m1_dict[output0_tensor_name]
    t2 = time.perf_counter() - t1
    print("Elapsed time: {}".format(t2))
    with open(build_dir+"/exec_base_result.npy", 'wb') as f:
        np.save(f, expected_m1)
        np.save(f, input_val)

    return input_val

def exec_hls_model():
    t1 = time.perf_counter()

    model_2 = ModelWrapper(build_dir+"/quartznet_hls.onnx")
    exec_mode="cppsim"

    if exec_mode=="cppsim":
        model_2 = model_2.transform(PrepareCppSim())
        model_2 = model_2.transform(CompileCppSim())
        model_2 = model_2.transform(SetExecMode("cppsim"))

    input_val = np.load(build_dir+"/brevitas_reference/end2end_quartznet_input_quantized.npy")
    input0_tensor_name = model_2.graph.input[0].name
    input_dict = {}
    input_val = np.reshape(input_val, np.shape(input_val)+(1,)) #extend to 4D
    input_dict[input0_tensor_name] = input_val
    output0_tensor_name = model_2.graph.output[0].name
    expected_m2_dict = oxe.execute_onnx(model_2, input_dict, return_full_exec_context = False)
    expected_m2 = expected_m2_dict[output0_tensor_name]

    #expected_m2 = np.reshape(expected_m2, np.shape(expected_m1))
    #m2_input_val = np.reshape(m2_input_val, np.shape(m1_input_val))

    t2 = time.perf_counter() - t1
    print("Elapsed time: {}".format(t2))
    with open(build_dir+"/exec_hlsmodel_threshadjusted_result.npy", 'wb') as f:
        np.save(f, expected_m2)
        np.save(f, input_val)

def execute_models():
    input_val = exec_base_model()
    exec_hls_model()
