import torch
import torch.nn.functional as F
import torch.optim as optim
from dataset.dataset import create_dl
import pickle
import random
from model_arch.decoder import Decoder
from argparse import ArgumentParser
from torchtune.modules import get_cosine_schedule_with_warmup
from pathlib import Path


def train(model: torch.nn.Module,
           loader: torch.utils.DataLoader,
             opt: optim.AdamW,
               epochs: int,
               device,
               tokenizer,
               grad_accum,
               scaler,
               output_dir,
               sched: get_cosine_schedule_with_warmup):
    padding_idx = tokenizer.tok_to_id_map["<PAD>"]
    model.train()
    glb_step = 0
    for epoch in range(epochs):
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(dtype=torch.bfloat16):
                logits = model(x).logits
                pred = logits.view(-1, logits.size(-1))
                target = y.view(-1)
                loss = F.cross_entropy(
                    pred, target, ignore_index=padding_idx) / grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.unscale(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched_step = sched.step()
                opt.zero_grad(set_to_none=True)
                glb_step+=1

                if glb_step % 100 == 0:
                    print(f'epoch{epoch + 1} | step {glb_step} | loss {loss.item():.4f}')
    ckpt_path = Path(output_dir) / f'model_epoch{epoch}.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f'saved model to {ckpt_path}')



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True, default = "slop_sample_1.txt", help="Plain‑text file (one doc per line)")
    parser.add_argument("--config", type=str, required=True, default = "model_arch/model_config.json", help="model_config.json (dense‑only for now)")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to saved tokenizer.pkl")
    parser.add_argument("--output_dir", type=str, default="ckpts_dense")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default= 2e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--optim_weight_decay", type=float, default=0.1)
    parser.add_argument("--optim_beta_1", type=float, default=0.9)
    parser.add_argument("--optim_beta_2", type=float, default=0.95)



    args = parser.parse_args()


    device = "cuda" if torch.cuda.is_available else "cpu"
    model = Decoder()
    dl = create_dl(args.corpus, args.block_size, args.batch_size)
    opt = optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.optim_weight_decay, betas=(args.optim_beta_1, args.optim_beta_2))
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps, num_training_steps=len(dl) * args.epochs // args.grad_accum)
    scaler = torch.amp.GradScaler(enabled=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()))
    criterion = torch.nn.CrossEntropyLoss()