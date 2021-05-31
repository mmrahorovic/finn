import os
import xml.etree.ElementTree as ET
from finn.custom_op import registry
from finn.util.fpgadataflow import is_fpgadataflow_node
from finn.core.modelwrapper import ModelWrapper

def set_estimated_clock(path):
    model = ModelWrapper(path)
    for n in model.graph.node:
        if is_fpgadataflow_node(n) is True:
            inst = registry.getCustomOp(n)
            code_gen_dir = inst.get_nodeattr("code_gen_dir_ipgen")
            if code_gen_dir=="":
                continue
            xmlfile = "{}/project_{}/sol1/syn/report/{}_csynth.xml".format(code_gen_dir, n.name, n.name)
            if os.path.isfile(xmlfile):
                tree = ET.parse(xmlfile)
                root = tree.getroot()
                for item in root.findall("PerformanceEstimates/SummaryOfTimingAnalysis"):
                    for child in item:
                        if child.tag=="unit":
                           unit = child.text
                        elif child.tag=="EstimatedClockPeriod":
                            estimated_clock = child.text
                for item in root.findall("PerformanceEstimates/SummaryOfOverallLatency"):
                    for child in item:
                        if child.tag=="unit":
                            unit_latency = child.text
                        if child.tag=="Best-caseLatency":
                            best_latency = child.text
                        if child.tag=="Average-caseLatency":
                            average_latency = child.text
                        if child.tag=="Worst-caseLatency":
                            worst_latency = child.text
                try:
                    result_clk = estimated_clock+" "+unit
                    result_latency = str([best_latency, average_latency, worst_latency])+" "+unit_latency
                except NameError:
                    print("Either clock or unit not found")
            inst.set_nodeattr("est_clk_hls", str(result_clk))
            inst.set_nodeattr("est_latency_hls", str(result_latency))
            modified = True

    if modified:
        #model.save(path)
        new_path = path.replace("hlssynth.onnx", "hlssynth_annotated.onnx")
        model.save(new_path)
