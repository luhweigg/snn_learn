import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate, functional

class CIFAR10_SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer.Flatten(),
            layer.Linear(2 * 32 * 32, 128, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(128, 10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor):
        functional.reset_net(self)
        out = self.network(x)
        return out.sum(dim=0)