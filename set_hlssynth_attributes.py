from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.util.fpgadataflow import is_fpgadataflow_node

def set_hlssynth_attributes(file, directory):
    model = ModelWrapper(file)

    for n in model.graph.node:
        if is_fpgadataflow_node(n):
            inst = getCustomOp(n)
            code_gen_dir = inst.get_nodeattr("code_gen_dir_ipgen")
            ipgen_path = code_gen_dir+"/project_{}".format(n.name)
            ip_path = ipgen_path+"/sol1/impl/ip"
            vlnv = "xilinx.com:hls:%s:1.0" % n.name
            inst.set_nodeattr("ipgen_path", ipgen_path)
            inst.set_nodeattr("ip_path", ip_path)
            inst.set_nodeattr("ip_vlnv", vlnv)

    new_file = file.replace(".onnx", "_hlssynth_attr.onnx")
    model.save(new_file)

