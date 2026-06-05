import torch
import torch.nn as nn

def quantize_weights(model: nn.Module, num_bits: int = 8):
    """
    Apply post-training quantization to the model's weights to simulate fixed-point arithmetic behavior of the FPGA.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                scale = param.abs().max() / ((2 ** (num_bits - 1)) - 1)
                param.copy_(torch.round(param / scale) * scale)