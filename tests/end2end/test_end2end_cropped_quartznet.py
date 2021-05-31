import time
import os

from finn.util.test import load_test_checkpoint_or_skip
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.replace_verilog_relpaths import ReplaceVerilogRelPaths
from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources
from finn.util.basic import alveo_part_map, alveo_default_platform

build_dir = os.environ["FINN_BUILD_DIR"]
test_board = "U250"
test_fpga_part = alveo_part_map[test_board]
target_clk_ns = 3

def test_end2end_cropped_quartznet_gen_hls_ip():
    model = load_test_checkpoint_or_skip(build_dir+"/end2end_quartznet_syn_v6.onnx")

    start = time.time()
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model.save(build_dir+"/end2end_quartznet_syn_v6_ip.onnx")
    model = model.transform(HLSSynthIP())
    model = model.transform(ReplaceVerilogRelPaths())
    model = model.transform(AnnotateResources("hls"))
    model.save(build_dir+"/end2end_quartznet_syn_v6_hlssynth.onnx")
    end = time.time()

    elapsed_time = end-start
    f = open(build_dir+"/end2end_quartznet_syn_v6_time.txt", "w+")
    f.write("Execution time in seconds: " + str(elapsed_time))
    f.close()
