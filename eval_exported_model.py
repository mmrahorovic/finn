import numpy as np
import torch
from brevitas_examples.speech_to_text.quartznet import model_with_cfg
from brevitas_examples.speech_to_text.quartznet.helpers import post_process_predictions
from brevitas.quant_tensor import QuantTensor
import brevitas.onnx as bo

vocab = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
         "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

def get_scale_zeropoint(x, num_of_bits=8):
    # https://leimao.github.io/article/Neural-Networks-Quantization/
    #alpha = x.min()
    #beta = x.max()
    alpha = -4.743273735046387
    beta = 29.181787490844727

    b = num_of_bits
    alpha_q = -2 ** (b-1)
    beta_q = 2 ** (b-1) - 1

    s = (beta - alpha) / (beta_q - alpha_q)
    z = int((beta * alpha_q - alpha * beta_q) / (beta - alpha))

    #x_q = torch.round(1 / s * x + z)
    #x_q = torch.clamp(x_q, min=alpha_q, max=beta_q)

    return s, z

def eval_exported_model():
    model, cfg = model_with_cfg('quant_quartznet_perchannelscaling_4b', pretrained=True, export_mode=True)
    model.eval()
    input_val = np.load("/workspace/results/librispeech_data/input_sample_float_0.npy")
    input_tensor = torch.from_numpy(input_val)

    ## QuantTensor
    bitwidth = 8
    s, z = get_scale_zeropoint(input_tensor, bitwidth)
    quant_tensor_input = QuantTensor(
        input_tensor,
        scale=torch.tensor(float(s)),
        zero_point=torch.tensor(float(z)),
        bit_width=torch.tensor(float(bitwidth)),
        signed=True,
        training=False
    )

    predictions = model(quant_tensor_input)

    greedy_hypotheses = post_process_predictions([predictions], vocab)
    print(greedy_hypotheses)

    #bo.export_finn_onnx(module=model, input_shape=input_val.shape, export_path="/tmp/exported_qn_model.onnx", input_t=quant_tensor_input)
    bo.export_finn_onnx(module=model, export_path="/tmp/exported_qn_model.onnx", input_t=quant_tensor_input)
