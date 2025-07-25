import requests
import regex as re
import torch.nn as nn
import torch
import torch.nn.functional as F

SLOP_HEURISTIC = re.compile(r"\X")
text = "😭""
re.finda(text, SLOP_HEURISTIC):
    print("match found")
else:
    print("no")



# class SlopDetector(nn.Module):


