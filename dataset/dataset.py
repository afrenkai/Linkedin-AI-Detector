import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Union
import random
from tokenization.build_vocab import BPETokenizer

class SlopDataset(Dataset):
    def __init__(self, file_path: str, block_size: int, tokenizer: Union[BPETokenizer, str], split: str = 'train', split_ratio: float = 0.9):
        self.block_size = block_size
        self.data: List[List[int]] = []
        if isinstance(tokenizer, str):
            self.tokenizer = BPETokenizer.load(tokenizer)
        else:
            self.tokenizer = tokenizer
            
        all_chunks = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids = self.tokenizer.encode(line.strip())
                    for i in range(0, len(ids) - block_size, block_size // 2): #~50% crossover?????
                        chunk = ids[i:i + block_size + 1]
                        if len(chunk) >= 2:
                            all_chunks.append(chunk)
        
        random.shuffle(all_chunks)
        split_idx = int(len(all_chunks) * split_ratio)
        if split == 'train':
            self.data = all_chunks[:split_idx]
        else: 
            self.data = all_chunks[split_idx:]
            
        print(f"Created {split} dataset with {len(self.data)} chunks")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        if len(seq) < self.block_size + 1:
            pad_token_id = self.tokenizer.tok_to_id_map.get("<PAD>", 0)
            seq = seq + [pad_token_id] * (self.block_size + 1 - len(seq))
        else:
            seq = seq[:self.block_size + 1]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y
    
def create_dl(corpus: str, block_size: int, batch_size: int, tokenizer: Union[BPETokenizer, str], split: str = 'train'):
    ds = SlopDataset(corpus, block_size, tokenizer, split)
    dl = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=(split == 'train'), 
        drop_last=True,
        num_workers=0 # TODO: not gonna fly on many gpus, research will be needed
    )
    return dl