import torch
import numpy as np
from models.generator import Generator

def test():
    input = torch.arange(0, 12, dtype=torch.long)
    print(input.shape)
    print(input)
    input = input.reshape(3, 4).to(torch.float32)
    print(input.shape)
    print(input)
    model = torch.nn.Conv2d(1, 1, (3, 1), (1, 1))
    output = model(input.unsqueeze(0).unsqueeze(0)) 
    print(output)
if __name__ == '__main__':
    test()