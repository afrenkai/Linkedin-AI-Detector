import optuna
import torch
import torch.nn.functional as F
import torch.optim as optim
import json
import pickle
import os
from pathlib import Path
from dataset.dataset import SlopDataset
from model_arch.SlopGPT import SlopGPT, SlopGPTConfig
from transformers.optimization import get_cosine_schedule_with_warmup
from train.evaluate import evaluate
from typing import Union
import logging
from tokenization.build_vocab import BPETokenizer
import argparse
from utils.consts import (JSON_INDENT, OBJECTIVE_EPOCHS, TUNING_TRAIN_RATIO, TUNING_EVAL_RATIO, MIN_CHUNK_SIZE, DEFAULT_PAD_TOKEN_ID)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlopDatasetTuning(SlopDataset):

    def __init__(self, corpus, block_size, tokenizer, split: str):
        super(SlopDataset).__init__(file_path = corpus, block_size = block_size, tokenizer  = tokenizer, split = split)
        self.block_size = block_size
        self.data = []
        
        if not os.path.exists(corpus):
            raise FileNotFoundError(f"Corpus file not found: {corpus}")
            
        if isinstance(tokenizer, str):
            self.tokenizer = BPETokenizer.load(tokenizer)
        else:
            self.tokenizer = tokenizer
            
        all_chunks = []
        with open(corpus, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids = self.tokenizer.encode(line.strip())
                    for i in range(0, len(ids) - block_size, block_size):
                        chunk = ids[i:i + block_size + 1]
                        if len(chunk) >= MIN_CHUNK_SIZE:
                            all_chunks.append(chunk)
        
        if not all_chunks:
            raise ValueError(f"No valid chunks found in {corpus}")

        train_end = int(len(all_chunks) * TUNING_TRAIN_RATIO)
        eval_end = int(len(all_chunks) * (TUNING_TRAIN_RATIO + TUNING_EVAL_RATIO))
        
        if split == 'train':
            self.data = all_chunks[:train_end]
        elif split == 'eval':
            self.data = all_chunks[train_end:eval_end]
        else:
            self.data = all_chunks[eval_end:]
            
        logger.info(f"Created {split} dataset with {len(self.data)} chunks")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        if len(seq) < self.block_size + 1:
            pad_token_id = self.tokenizer.tok_to_id_map.get("<PAD>", DEFAULT_PAD_TOKEN_ID)
            seq = seq + [pad_token_id] * (self.block_size + 1 - len(seq))
        else:
            seq = seq[:self.block_size + 1]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

def create_dl_tuning(corpus: str, block_size: int, batch_size: int, tokenizer, split: str = 'train'):
    ds = SlopDatasetTuning(corpus, block_size, tokenizer, split)
    dl = torch.utils.data.DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=(split == 'train'), 
        drop_last=True,
        num_workers=0
    )
    return dl

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for hyperparameter optimization"""

    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [2, 4, 8, 16]),
        'warmup_steps': trial.suggest_int('warmup_steps', 100, 2000, step=100),
        'grad_accum': trial.suggest_categorical('grad_accum', [4, 8, 16, 32]),
        'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.2),
        'beta_1': trial.suggest_float('beta_1', 0.8, 0.95),
        'beta_2': trial.suggest_float('beta_2', 0.9, 0.999),
        'block_size': trial.suggest_categorical('block_size', [512, 1024, 2048]),
        'hidden_size': trial.suggest_categorical('hidden_size', [1024, 2048, 4096]),
        'num_heads': trial.suggest_categorical('num_heads', [8, 16, 32]),
        'num_layers': trial.suggest_categorical('num_layers', [6, 12, 24]),
        'attention_dropout': trial.suggest_float('attention_dropout', 0.0, 0.2),
    }

    corpus = "slop_sample_1.txt"
    config_path = "model_arch/model_config.json"
    tokenizer_path = "tokenizer.pkl"
    epochs = OBJECTIVE_EPOCHS 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:

        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        

        with open(config_path, 'r') as f:
            config_data = json.load(f)

        config_data.update({
            'hidden_size': params['hidden_size'],
            'num_attention_heads': params['num_heads'],
            'num_hidden_layers': params['num_layers'],
            'attention_dropout': params['attention_dropout'],
            'block_size': params['block_size']
        })
        
        config = SlopGPTConfig(**config_data)
        model = SlopGPT(config).to(device)


        #mypy false positives here since these are references to optuna trial suggestions. however, good note for general type checking

        train_dl = create_dl_tuning(corpus = corpus, block_size = params['block_size'], batch_size= params['batch_size'], tokenizer = tokenizer, split = 'train')
        eval_dl = create_dl_tuning(corpus = corpus, block_size = params['block_size'], batch_size = params['batch_size'], tokenizer = tokenizer, split = 'eval')

        opt = optim.AdamW(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay'],
            betas=(params['beta_1'], params['beta_2'])
        )
        
        total_steps = len(train_dl) * epochs // params['grad_accum']
        sched = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=params['warmup_steps'],
            num_training_steps=total_steps
        )

        scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))

        padding_idx = tokenizer.tok_to_id_map["<PAD>"]
        best_eval_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for step, (x, y) in enumerate(train_dl):
                x, y = x.to(device), y.to(device)
                
                device_type = 'cuda' if device.startswith('cuda') else 'cpu'
                with torch.amp.autocast(device_type=device_type, enabled=True, dtype=torch.bfloat16):
                    logits, _ = model(x)
                    pred = logits.view(-1, logits.size(-1))
                    target = y.view(-1)
                    loss = F.cross_entropy(pred, target, ignore_index=padding_idx) / params['grad_accum']
                
                scaler.scale(loss).backward()
                epoch_loss += loss.item() * params['grad_accum']
                num_batches += 1
                
                if (step + 1) % params['grad_accum'] == 0:
                    scaler.unscale_(opt)
                    #TODO: come back to this !!!!!!!!
                    # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP_NORM)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad(set_to_none=True)

            eval_loss = evaluate(model, eval_dl, tokenizer, device)
            
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss

            trial.report(eval_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_eval_loss
        
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        return float('inf')

def run_hyperparameter_optimization(
    n_trials: int = 50,
    study_name: str = "slop_gpt_optimization",
    storage: Union[str, None] = None
) -> optuna.Study:
    

    if storage:
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
    else:
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name
        )
    
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
    logger.info(f"Study name: {study_name}")

    study.optimize(objective, n_trials=n_trials)
    
    logger.info("Best trial:")
    trial = study.best_trial
    
    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    
    return study

def save_optimization_results(study: optuna.Study, output_dir: str = "tuning_results"):
   
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    best_params = study.best_trial.params
    with open(output_path / "best_params.json", 'w') as f:
        json.dump(best_params, f, indent=JSON_INDENT)
    
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            duration = trial.duration.total_seconds() if trial.duration else 0.0
            trials_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'duration': duration
            })
    
    with open(output_path / "all_trials.json", 'w') as f:
        json.dump(trials_data, f, indent=JSON_INDENT)
    
    logger.info(f"Results saved to {output_path}")
    return best_params

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for SlopGPT")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--study_name", type=str, default="slop_gpt_optimization", help="Study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL")
    parser.add_argument("--output_dir", type=str, default="tuning_results", help="Output directory")
    
    args = parser.parse_args()
    
    study = run_hyperparameter_optimization(
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage
    )

    best_params = save_optimization_results(study, args.output_dir)
    
    logger.info("Hyperparameter optimization completed!")
    logger.info(f"Best validation loss: {study.best_value:.4f}") 