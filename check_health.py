import platform
import torch

def test_torch() -> str:
    """
    pretty much self explanatory, but just in case:
    if linux: check cuda
    if mac: check mps
    """
    if platform.system() == "Linux":
        if torch.cuda.is_available() == 1:
            return "Cuda detected, yes!"
        else:
            return "No Cuda Detected"
    elif platform.system() == "Darwin":
        if torch.backends.mps.is_available() == 1:
            return "MPS detected, yes!"
        else:
            return "No MPS"
    return "No MPS or Cuda Found, defaulting to cpu"
