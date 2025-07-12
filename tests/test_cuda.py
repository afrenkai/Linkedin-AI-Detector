import torch
import numpy as np
def test_cuda():
    assert(torch.cuda.is_available() == 1)