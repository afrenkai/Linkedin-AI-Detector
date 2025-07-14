import torch
import torch.nn.functional as F
import torch.optim as optim
import json
from typing import Union
import pickle
from dataset.dataset import create_dl
from model_arch.SlopGPT import SlopGPT, SlopGPTConfig
from argparse import ArgumentParser
from torchtune.training import get_cosine_schedule_with_warmup
from pathlib import Path
from train.train_utils import save_checkpoint, setup_logging, save_tokenizer
from train.evaluate import evaluate
from tokenization.build_vocab import BPETokenizer

def train(model: torch.nn.Module,
           loader: torch.utils.data.DataLoader,
           val_loader: torch.utils.data.DataLoader,
           opt: optim.AdamW,
           epochs: int,
           device,
           tokenizer: BPETokenizer,
           grad_accum,
           scaler,
           output_dir,
           sched,
           logger,
           start_epoch: int = 0,
           global_step: int = 0):

    padding_idx = tokenizer.tok_to_id_map["<PAD>"]
    model.train()
    best_val_loss = float('inf')
    

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting training from epoch {start_epoch}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast(device_type='cuda', enabled = True, dtype=torch.bfloat16):
                logits, _ = model(x)
                pred = logits.view(-1, logits.size(-1))
                target = y.view(-1)
                loss = F.cross_entropy(pred, target, ignore_index=padding_idx) / grad_accum
            
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum
            num_batches += 1
            
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 100 == 0:
                    current_lr = sched.get_last_lr()[0]
                    log_msg = f'Epoch {epoch + 1} | Step {global_step} | Loss {loss.item() * grad_accum:.4f} | LR {current_lr:.2e} | Grad Norm {grad_norm:.4f}'
                    logger.info(log_msg)
                    
        
        avg_train_loss = epoch_loss / num_batches
        val_loss = evaluate(model, val_loader, tokenizer, device)
        
        logger.info(f'Epoch {epoch + 1} completed - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
        ckpt_path = Path(output_dir) / f'model_epoch_{epoch + 1}.pt'
        save_checkpoint(model, opt, scaler, sched, epoch, global_step, ckpt_path)
        logger.info(f'Saved checkpoint to {ckpt_path}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = Path(output_dir) / 'best_model.pt'
            save_checkpoint(model, opt, scaler, sched, epoch, global_step, best_ckpt_path)
            logger.info(f'New best model saved with val loss {val_loss:.4f}')
    
    logger.info("Training completed!")
    return global_step



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True, default="slop_sample_1.txt", help="Plain‑text file (one doc per line)")
    parser.add_argument("--config", type=str, required=True, default="model_arch/model_config.json", help="model_config.json (dense‑only for now)")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to saved tokenizer.pkl")
    parser.add_argument("--output_dir", type=str, default="ckpts_dense")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--optim_weight_decay", type=float, default=0.1)
    parser.add_argument("--optim_beta_1", type=float, default=0.9)
    parser.add_argument("--optim_beta_2", type=float, default=0.95)
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--eval_every", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N steps")

    args = parser.parse_args()

    logger = setup_logging(args.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    with open(args.config, 'r') as f:
        config_data = json.load(f)
    config = SlopGPTConfig(**config_data)
    model = SlopGPT(config).to(device)

    if args.tokenizer.isinstance(BPETokenizer):
        logger.info("loading unplickled tokenizer, I would recommend pickling this. I'm probably lazy and haven't written the method to load/save tokenizers yet.")
        tokenizer = args.tokenizer
    else:
        with open(args.tokenizer, 'rb') as f:
            tokenizer = pickle.load(f)
        logger.info(f"Loaded tokenizer with vocab size: {len(tokenizer.tok_to_id_map)}")
    
    train_dl = create_dl(args.corpus, args.block_size, args.batch_size, tokenizer, split='train')
    val_dl = create_dl(args.corpus, args.block_size, args.batch_size, tokenizer, split='val')

    opt = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.optim_weight_decay, 
        betas=(args.optim_beta_1, args.optim_beta_2)
    )
    
    total_steps = len(train_dl) * args.epochs // args.grad_accum
    sched = get_cosine_schedule_with_warmup(
        opt, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps
    )
    
    scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))


    start_epoch = 0
    global_step = 0
    if args.resume_path:
        logger.info(f"Resuming from checkpoint: {args.resume_path}")
        ckpt = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_tokenizer(tokenizer, output_path / "tokenizer.pkl")
    
    with open(output_path / "config.json", 'w') as f:
        json.dump(config_data, f, indent=2)
    
    with open(output_path / "training_args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("Saved tokenizer, config, and training args")

    logger.info("Starting training...")
    final_step = train(
        model=model,
        loader=train_dl,
        val_loader=val_dl,
        opt=opt,
        epochs=args.epochs,
        device=device,
        tokenizer=tokenizer,
        grad_accum=args.grad_accum,
        scaler=scaler,
        output_dir=args.output_dir,
        sched=sched,
        logger=logger,
        start_epoch=start_epoch,
        global_step=global_step
    )
    
    
    logger.info(f"Training completed at step {final_step}")