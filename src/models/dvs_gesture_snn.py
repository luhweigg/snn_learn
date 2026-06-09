import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate, functional

class DVSGesture_SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
            layer.Dropout2d(0.1),
            
            layer.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
            layer.Dropout2d(0.15),
            
            layer.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(256),
            neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
            layer.Dropout2d(0.2),
            
            layer.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(256),
            neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
            layer.Dropout2d(0.25),
            
            layer.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(512),
            neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),
            layer.Dropout2d(0.3),
            
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(512 * 4 * 4, 11, bias=True),
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor):
        functional.reset_net(self)
        out_spikes = self.network(x)
        return out_spikes.mean(dim=0)