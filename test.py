import torch
import numpy as np
from models.generator import Generator

def test():
    input = torch.randn(1, 80, 100)
    model = Generator(80)
    model.num_params()
    output = model(input)
    print(output.shape)
    
if __name__ == '__main__':
    test()