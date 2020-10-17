import torch
import torch.nn as nn
import numpy as np
from models.layers import MRF, UpsampleNet


class Generator(nn.Module):
    def __init__(self, 
                 in_channels):
        super().__init__()
        self.generator = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(in_channels, 512, kernel_size=7)),
            nn.LeakyReLU(0.2),
            UpsampleNet(512, 256, 8),
            MRF(256),
            nn.LeakyReLU(0.2),
            UpsampleNet(256, 128, 8),
            MRF(128),
            nn.LeakyReLU(0.2),
            UpsampleNet(128, 64, 2),
            MRF(64),
            nn.LeakyReLU(0.2),
            UpsampleNet(64, 32, 2),
            MRF(32),
            nn.ReflectionPad1d(3),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(32, 1, kernel_size=7)),
            nn.Tanh(),
        )

    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)

    def remove_weight_norm(self):

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
                
        self.apply(_remove_weight_norm)

    def forward(self, inputs):
        return self.generator(inputs)
