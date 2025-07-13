import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
from tokenization.build_vocab import BPETokenizer

class SlopDataset(Dataset):
    def __init__(self, file_path: str, block_size: int):
        self.block = block_size
        self.data: List[int] = []
        with open (file_path, encoding="utf-8") as f:
            for line in f:
                ids = BPETokenizer().encode(line)
                for i in range(0, len(ids) -1, block_size):
                    chunk = ids[i: i + block_size + 1]
                if len(chunk) >= 2:
                    self.data.append(chunk)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        x = torch.tensor(seq[:-1], dtype = torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y
    
def create_dl(corpus: str, block_size: int, batch_size: int, split: str = 'train'):
    # TODO: padding?
    ds = SlopDataset(corpus, block_size)
    dl = DataLoader(ds, batch_size, shuffle = True if split == 'train' else False, drop_last=True)
    return dl