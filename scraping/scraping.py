import re
import pandas as pd
import os
import torch.nn as nn
import torch
from tokenization.build_vocab import BPETokenizer
from utils.consts import DEFAULT_PAD_TOKEN_ID
from torch.nn.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def test_slop(text, heuristic) -> dict:
    detected_chars = {}
    for ch in text:
        if re.match(heuristic, ch):
            detected_chars[ch] = text.index(ch)
            print(f"match found: {ch} at {text.index(ch)}")
        else:
            continue
    if len(detected_chars.keys()) >= 3 or  "—" in detected_chars.keys():
        print("yea ts slop gng throw ts in 🥀")
    
    return detected_chars



# def test_method(emoji):
#     if re.match(r"\, emoji):
#         print("match found")
#     else:
#         print("no")
#

# current thought process is something like:
# train a small classifier to detect linkedin slop beforehand to trim samples for "personality"
# fineweb for actual knowledge base. optuna for hparam opt, train, test, eval,try all loss_fns, etc.
#slop sample would be more simclr with labels like starmie, not versions like my paper
#based on my research and starmie

class ScrapedDataset(Dataset):
    def __init__(self,
                 slop_samples_path,
                 max_seq_len : int,
                 tokenizer: BPETokenizer):
        self.slop_samples_path = slop_samples_path
        self.tokenizer = tokenizer
        self.pairs = []
        self.max_seq_len = max_seq_len
        self.samples = pd.read_csv(self.slop_samples_path)
        self.labels = self.samples['slop']
        self.table_cache = {}

    def _read_slop_sample(self, sample_id):
        """Read a sample"""
        if sample_id in self.table_cache:
            slop_sample = self.table_cache[sample_id]
        else:
            slop_sample = pd.read_csv(os.path.join(self.slop_samples_path,
                                             "sample.csv" % sample_id))
            self.table_cache[sample_id] = slop_sample

        return slop_sample


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return a tokenized and augmented item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token IDs of the two entities combined
            int: the label of the pair (0: not slop, 1: slop)
        """
        # idx = random.randint(0, len(self.pairs)-1)
        base_slop = self.samples[idx]

        if len(base_slop) > self.max_seq_len + 1:
            pad_token_id = self.tokenizer.tok_to_id_map.get("<PAD>", DEFAULT_PAD_TOKEN_ID)
            base_slop = base_slop + [pad_token_id] * (self.max_seq_len + 1 - len(base_slop))
        else:
            base_slop = base_slop[:self.max_seq_len + 1]

        X = torch.tensor(base_slop[:-1], dtype = torch.long)
        y = torch.tensor(base_slop[:1], dtype = torch.long)
        return X, y

def create_slop_dl(path: str, max_seq_len: int, batch_size: int, tokenizer: BPETokenizer, device: str) -> DataLoader:
    ds = ScrapedDataset(path, max_seq_len, tokenizer)
    dl = DataLoader(
        ds, 
        batch_size,
        device = device, 
        num_workers = 0

    )
    return dl





class SlopDetector(nn.Module):
    """causal self-attn as per nano-gpt"""
    def __init__(self, n_heads, bias: bool, embed_dim:int, causal: bool,dropout: float) -> None:
        super().__init__()
        assert embed_dim % n_heads == 0
        self.causal_attn = nn.Linear(embed_dim, 3*embed_dim, bias=bias)
        self.causal_prog = nn.proj(embed_dim, embed_dim)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.causal = causal

    def forward(self, x):
        q_proj  = self.causal_attn(x)
        bs = q_proj.size(0)
        embed_dim = q_proj.size(2)
        h_dim = embed_dim//(self.n_heads*3)

        q, k, v = q_proj.chunk(3, -1)
        q = q.view(bs, -1, self.n_heads, h_dim).transpose(1, 2)
        k = k.view(bs, -1, self.n_heads, h_dim).transpose(1, 2)
        v = v.view(bs, -1, self.n_heads, h_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
            causal = self.causal
        else:
            dropout = 0.0
            causal = False

        y = F.scaled_dot_product_attention(q, k, v, attn_mask = None, dropout_p = dropout, is_causal = causal)
        y = y.transpose(1, 2).view(bs, -1, self.n_heads * h_dim)
        return y



#
# if __name__ == "__main__":
#     text = """
#
#     Just gave a demo of CLAUDE code to a client. ✨
#
#     I initially was worried about what to show, but when I ran claude I knew exactly what to do.
#
#     Idea: Just finish off one of the tasks from my todo list, with full automation.
#
#     Fetch task -> Planning (with plan mode) -> Walk through todo list -> Run tests -> Make a PR -> Gemini code assist did code review -> Custom claude command to fetch reviews and fix -> Update the PR with description
#
#     We summed up my 2 hour of work items, in 15 mins of demo. ✅
#
#     Is it worth the price? Maybe.
#     Is it a lot cheaper than an intern would cost and does a better job at it? Absolutely Yes.
#     Are we at a point where dev jobs are lost? Hell no.
#
#     We need people who are educated on this to get better results faster. No team should wait 3 months to launch a product without knowing it's product-market-fit.
#
#     So let's see what future has in store.
#
#     I'm trying out GEMINI CLI next, let me know what I should build next. 💪
#
#     """
#     # print(test_slop(text, SLOP_HEURISTIC))
#     n_heads = 8
#     hpd = 64
#     embed_dim = n_heads * hpd
#     dtype = torch.bfloat16
#     model = SlopDetector(n_heads, bias = True, embed_dim = embed_dim, causal = True, dropout = 0.1)
#     print(model)
#
#

