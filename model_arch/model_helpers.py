
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor




class SwiGLU(nn.Module):
    """
    swish-gated linear unit

    2 affine functions interspersed by a siwsh function, with beta = 1
    similar to pretty much any other gated residual connection, just happens to work

    'We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence.'
    - the dudes that made it

    Swiglu(x, w, v, b, cbeta) = swish( xW + b) X (xV + c) = 
    """
    def __init__(self, dim : int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, 2*dim)

    def forward(self, x:Tensor):
        x1, x2 = self.proj(x).chunk(2, dim =-1)
        return x1 * F.silu(x2)

STR_TO_FN_MAP = {"swiglu": SwiGLU, "gelu": nn.GELU, "silu" : nn.SiLU, "relu": nn.ReLU}
