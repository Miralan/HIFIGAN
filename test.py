import torch
import numpy as np
from models.generator import Generator

def test():
    input = torch.arange(0, 12, dtype=torch.long)
    print(input.shape)
    print(input)
    # input = input.reshape(3, 4).to(torch.float32).unsqueeze(0).permute(0, 2, 1)
    input = input.view(3, 4)
    print(input.shape)
    print(input)
if __name__ == '__main__':
    test()