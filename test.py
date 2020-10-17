import torch
import numpy as np
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, 
                in_channels, 
                nums, 
                kernel_size, 
                dialations):

        super().__init__()
        print(nums)
        self.layer = nn.Sequential(
            *[
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(in_channels, in_channels, kernel_size, 1, dilation=dialations[i]
                              , padding=(kernel_size - 1)*dialations[i] // 2)
                )
                for i in range(nums)
            ]
        )

    def forward(self, x):
        return x + self.layer(x)


class MRF(nn.Module):
    def __init__(self, 
                 in_channels, 
                 num = [3, 3], 
                 kernel_sizes = [3, 7, 11], 
                 dialations = [[[1, 1], [3, 1], [5, 1]] for i in range(3)]):

        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                *[ResBlock(in_channels, 2, kernel_sizes[i], dialations[i][j]) for j in range(num[0])]
                ) for i in range(num[1])
            ]
        )

    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)

    def forward(self, x):
        return torch.sum(torch.stack([layer(x) for layer in self.layers]), dim=0)

model = MRF(10)
print(model)