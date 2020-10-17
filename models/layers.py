import torch
import numpy as np
import torch.nn as nn


class ResStack(nn.Module):
    def __init__(self, channel, kernel_size, dilations):
        super(ResStack, self).__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(),
                nn.utils.weight_norm(nn.Conv1d(channel, channel,
                    kernel_size=kernel_size, dilation=dilations[0], padding=(kernel_size - 1)*dilations[0] // 2)),
                nn.LeakyReLU(),
                nn.utils.weight_norm(nn.Conv1d(channel, channel,
                    kernel_size=kernel_size, dilation=dilations[1], padding=(kernel_size - 1)*dilations[1] // 2)),
            )
            for i in range(3)
        ])

    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

class MRF(nn.Module):
    def __init__(self, 
                 channel,  
                 kernel_sizes = [3, 7, 11], 
                 dilations = [[1, 1], [3, 1], [5, 1]]):

        super().__init__()
        self.layers = nn.ModuleList(
            [
                ResStack(channel, kernel_sizes[0], dilations[0]),
                ResStack(channel, kernel_sizes[1], dilations[1]),
                ResStack(channel, kernel_sizes[2], dilations[2]),
            ]
        )

    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)

    def forward(self, x):
        return torch.sum(torch.stack([layer(x) for layer in self.layers]), dim=0)


class UpsampleNet(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 upsample_factor):

        super(UpsampleNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.upsample_factor = upsample_factor

        layer = nn.ConvTranspose1d(input_size, output_size, upsample_factor * 2,
                                   upsample_factor, padding=upsample_factor // 2)
        self.layer = nn.utils.weight_norm(layer)

    def forward(self, inputs):
        outputs = self.layer(inputs)
        outputs = outputs[:, :, : inputs.size(-1) * self.upsample_factor]
        return outputs
