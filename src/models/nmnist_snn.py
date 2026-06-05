import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate, functional

class NMNIST_SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer.Flatten(),
            layer.Linear(2 * 34 * 34, 256, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.Dropout(0.5),
            layer.Linear(256, 10, bias=False),
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor):
        functional.reset_net(self)
        out_spikes = self.network(x)
        return out_spikes.sum(dim=0)