import json
from model_arch.SlopGPT import SlopGPT, SlopGPTConfig
import torch

def test_logits():
    with open ('model_arch/model_config.json', 'rb') as f:
        data = json.load(f)
        device = 'cuda' if torch.cuda.is_available else 'cpu'
        config = SlopGPTConfig(**data)
        tokens = torch.randint(0, config.vocab_size, (2, 128)).to(device)
        model = SlopGPT(config).to(device)
        logits, _ = model(tokens)
        assert logits.shape == (tokens.size(0), tokens.size(1), config.vocab_size) # tensor of shape [[[batch_size]], [[seq_len]], [[vocab_size]]]


