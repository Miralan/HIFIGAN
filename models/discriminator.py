import torch
import torch.nn as nn
import numpy as np


class MPD(nn.Module):
    def __init__(self, 
                 p,
                 in_length):
        super().__init__()
        self.p = p
        self.period = p - in_length % p
        if self.period  % 2 != 0:
            self.pad = nn.ReflectionPad1d((self.period // 2, self.period  // 2 + 1))
            self.len = (in_length + self.period + 1) // self.p
        else:
            self.pad = nn.ReflectionPad1d(self.period // 2)
            self.len = (in_length + self.period) // self.p

        self.conv_layers = nn.Sequential(
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(1, 32, kernel_size=(5, 1), 
                                               stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(32, 128, kernel_size=(5, 1), 
                                               stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(128, 512, kernel_size=(5, 1), 
                                               stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(512, 1024, kernel_size=(5, 1), 
                                               stride=(1, 1), padding=(2, 0))),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(1024, 1024, kernel_size=(5, 1), 
                                               stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(0.2)
            ),
            nn.utils.weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), 
                                           stride=(1, 1), padding=(1, 0)))          
        )

    def forward(self, inputs):
        x = self.pad(inputs)
        assert x.shape[-1] == self.p * self.len
        x = x.reshape(x.shape[0], 1, self.len, self.p)
        x = self.conv_layers(x)
        return x


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, 
                 periods=(2, 3, 5, 7, 11), 
                 in_length=256*100):
        super().__init__()
        self.discriminators = nn.ModuleList([
            MPD(period, in_length) for period in periods
        ])

    def num_params(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)

    def forward(self, x):
        outputs = []

        for model in self.discriminators:
            output = model(x)
            outputs.append(output)
        return outputs


class MSD(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                nn.utils.weight_norm(nn.Conv1d(1, 16, kernel_size=15)),
                nn.LeakyReLU(0.2, True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(16, 64, kernel_size=41,
                                     stride=4, padding=20, groups=4)),
                nn.LeakyReLU(0.2, True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(64, 256, kernel_size=41,
                                     stride=4, padding=20, groups=16)),
                nn.LeakyReLU(0.2, True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(256, 512, kernel_size=41,
                                     stride=4, padding=20, groups=64)),
                nn.LeakyReLU(0.2, True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(512, 512, kernel_size=5,
                                     stride=1, padding=2)),
                nn.LeakyReLU(0.2, True),
            ),
            nn.utils.weight_norm(nn.Conv1d(512, 1, kernel_size=3,
                                 stride=1, padding=1)),
        ])

    def forward(self, x):
        for layer in self.discriminator:
            x = layer(x)

        return x 

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [MSD() for _ in range(3)]
        )
        
        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)

    def num_params(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)

    def forward(self, x):
        outputs = []

        for layer in self.discriminators:
            output = layer(x)
            outputs.append(output)

            x = self.downsample(x)

        return outputs

class Discriminator(nn.Module):
    def __init__(self, in_length):
        super().__init__()
        self.MSDs = MultiScaleDiscriminator()
        self.MPDs = MultiPeriodDiscriminator(in_length=in_length)

    def num_params(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)

    def forward(self, x):
        outputs1 = self.MSDs(x)
        outputs2 = self.MPDs(x)
        return outputs1, outputs2

# For unit network test
def test():
    model = Discriminator(25600)
    model.num_params()


if __name__ == '__main__':
    test()
