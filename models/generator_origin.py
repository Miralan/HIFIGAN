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
        self.layer = nn.Sequential(
            *[
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.utils.weight_norm(nn.Conv1d(in_channels, in_channels, kernel_size, 1, dilation=dialations[i]
                              , padding=(kernel_size - 1)*dialations[i] // 2))
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
        

def test():
    input = torch.randn(1, 80, 100)
    model = Generator(80)
    model.num_params()
    output = model(input)
    print(output.shape)
    

if __name__ == '__main__':
    test()