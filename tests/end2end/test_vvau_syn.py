import os

from finn.core.modelwrapper import ModelWrapper
from finn.transformation.general import ApplyConfig
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext
from finn.util.basic import alveo_part_map, alveo_default_platform

build_dir = os.environ["FINN_BUILD_DIR"]
test_board = "U250"
test_platform = alveo_default_platform[test_board]
test_fpga_part = alveo_part_map[test_board]
target_clk_ns = 6

def run_ooc():
    model = ModelWrapper(build_dir+"/vvau20.onnx")

    print("Running ApplyConfig...")
    model = model.transform(ApplyConfig(build_dir+"/fold_config.json"))
    model.save(build_dir+"/vvau20_config.onnx")

    print("Running PrepareIP...")
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model.save(build_dir+"/vvau20_prepareip.onnx")

    print("Running HLSSynthIP...")
    model = model.transform(HLSSynthIP())
    model.save(build_dir+"/vvau20_hlssynthip.onnx")
