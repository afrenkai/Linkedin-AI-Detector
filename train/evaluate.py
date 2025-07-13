import torch
import torch.nn.functional as F

def evaluate(model, val_loader, tokenizer, device):
    model.eval()
    total_loss = 0
    padding_idx = tokenizer.tok_to_id_map["<PAD>"]

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(dtype=torch.bfloat16):
                logits, _= model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=padding_idx)
            total_loss += loss.item()
    model.train()
    return total_loss / len(val_loader)