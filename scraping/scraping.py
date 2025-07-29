import re
import pandas as pd
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from tokenization.build_vocab import BPETokenizer
from utils.consts import DEFAULT_BLOCK_SIZE, DEFAULT_PAD_TOKEN_ID, SLOP_HEURISTIC


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
#     print(test_slop(text, SLOP_HEURISTIC))
#
# current thought process is something like:
# train a small classifier to detect linkedin slop beforehand to trim samples for "personality"
# fineweb for actual knowledge base. optuna for hparam opt, train, test, eval,try all loss_fns, etc.
#slop sample would be more simclr with labels like starmie, not versions like my paper
#based on my research and starmie

class ScrapedDataset:
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
            List of int: token ID's of the two entities combined
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

def create_slop_dl(path: str, max_seq_len: int, batch_size: int, tokenizer: Union[BPETokenizer, str], device: str) -> DataLoader:
    ds = ScrapedDataset(path, max_seq_len, tokenizer)
    dl = DataLoader(
        ds, 
        batch_size,

    )



  



def nxent_

class PretrainTableDataset(data.Dataset):
    """Table dataset for pre-training"""

    def __init__(self,
                 path,
                 augment_op,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 single_column=False,
                 sample_meth='wordProb',
                 table_order='column'):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.max_len = max_len
        self.path = path

        self.tables = [fn for fn in os.listdir(path) if '.csv' in fn]
        if size is not None:
            self.tables = self.tables[:size]

        self.table_cache = {}
        self.log_cnt = 0

        self.tokenizer_cache = {}
    # @staticmethod
    # def from_hp(path: str, hp: Namespace):
    #     """Construct a PretrainTableDataset from hyperparameters
    #
    #     Args:
    #         path (str): the path to the table directory
    #         hp (Namespace): the hyperparameters
    #
    #     Returns:
    #         PretrainTableDataset: the constructed dataset
    #     """
    #     return PretrainTableDataset(path,
    #                      augment_op=hp.augment_op,
    #                      lm=hp.lm,
    #                      max_len=hp.max_len,
    #                      size=hp.size,
    #                      single_column=hp.single_column,
    #                      sample_meth=hp.sample_meth,
    #                      table_order=hp.table_order)
    #
    #
    # def _read_table(self, table_id):
    #     """Read a table"""
    #     if table_id in self.table_cache:
    #         table = self.table_cache[table_id]
    #     else:
    #         fn = os.path.join(self.path, self.tables[table_id])
    #         table = pd.read_csv(fn, lineterminator='\n')
    #         self.table_cache[table_id] = table
    #
    #     return table
    #
    #
    # def _tokenize(self, table: pd.DataFrame) -> List[int]:
    #     """Tokenize a DataFrame table
    #
    #     Args:
    #         table (DataFrame): the input table
    #
    #     Returns:
    #         List of int: list of token ID's with special tokens inserted
    #         Dictionary: a map from column names to special tokens
    #     """
    #     res = []
    #     max_tokens = self.max_len * 2 // len(table.columns)
    #     budget = max(1, self.max_len // len(table.columns) - 1)
    #     tfidfDict = computeTfIdf(table) if "tfidf" in self.sample_meth else None # from preprocessor.py
    #
    #     # a map from column names to special token indices
    #     column_mp = {}
    #
    #     # column-ordered preprocessing
    #     if self.table_order == 'column':
    #         if 'row' in self.sample_meth: 
    #             table = tfidfRowSample(table, tfidfDict, max_tokens)
    #         for column in table.columns:
    #             tokens = preprocess(table[column], tfidfDict, max_tokens, self.sample_meth) # from preprocessor.py
    #             col_text = self.tokenizer.cls_token + " " + \
    #                     ' '.join(tokens[:max_tokens]) + " "
    #
    #             column_mp[column] = len(res)
    #             res += self.tokenizer.encode(text=col_text,
    #                                     max_length=budget,
    #                                     add_special_tokens=False,
    #                                     truncation=True)
    #     else:
    #         # row-ordered preprocessing
    #         reached_max_len = False
    #         for rid in range(len(table)):
    #             row = table.iloc[rid:rid+1]
    #             for column in table.columns:
    #                 tokens = preprocess(row[column], tfidfDict, max_tokens, self.sample_meth) # from preprocessor.py
    #                 if rid == 0:
    #                     column_mp[column] = len(res)
    #                     col_text = self.tokenizer.cls_token + " " + \
    #                             ' '.join(tokens[:max_tokens]) + " "
    #                 else:
    #                     col_text = self.tokenizer.pad_token + " " + \
    #                             ' '.join(tokens[:max_tokens]) + " "
    #
    #                 tokenized = self.tokenizer.encode(text=col_text,
    #                                     max_length=budget,
    #                                     add_special_tokens=False,
    #                                     truncation=True)
    #
    #                 if len(tokenized) + len(res) <= self.max_len:
    #                     res += tokenized
    #                 else:
    #                     reached_max_len = True
    #                     break
    #
    #             if reached_max_len:
    #                 break
    #
    #     self.log_cnt += 1
    #     if self.log_cnt % 5000 == 0:
    #         print(self.tokenizer.decode(res))
    #
    #     return res, column_mp
    #
    #
    # def __len__(self):
    #     """Return the size of the dataset."""
    #     return len(self.tables)
    #
    # def __getitem__(self, idx):
    #     """Return a tokenized item of the dataset.
    #
    #     Args:
    #         idx (int): the index of the item
    #
    #     Returns:
    #         List of int: token ID's of the first view
    #         List of int: token ID's of the second view
    #     """
    #     table_ori = self._read_table(idx)
    #
    #     # single-column mode: only keep one random column
    #     if self.single_column:
    #         col = random.choice(table_ori.columns)
    #         table_ori = table_ori[[col]]
    #
    #     # apply the augmentation operator
    #     if ',' in self.augment_op:
    #         op1, op2 = self.augment_op.split(',')
    #         table_tmp = table_ori
    #         table_ori = augment(table_tmp, op1)
    #         table_aug = augment(table_tmp, op2)
    #     else:
    #         table_aug = augment(table_ori, self.augment_op)
    #
    #     # convert table into string
    #     x_ori, mp_ori = self._tokenize(table_ori)
    #     x_aug, mp_aug = self._tokenize(table_aug)
    #
    #     # make sure that x_ori and x_aug has the same number of cls tokens
    #     # x_ori_cnt = sum([int(x == self.tokenizer.cls_token_id) for x in x_ori])
    #     # x_aug_cnt = sum([int(x == self.tokenizer.cls_token_id) for x in x_aug])
    #     # assert x_ori_cnt == x_aug_cnt
    #
    #     # insertsect the two mappings
    #     cls_indices = []
    #     for col in mp_ori:
    #         if col in mp_aug:
    #             cls_indices.append((mp_ori[col], mp_aug[col]))
    #
    #     return x_ori, x_aug, cls_indices
    #
    #
    # def pad(self, batch):
    #     """Merge a list of dataset items into a training batch
    #
    #     Args:
    #         batch (list of tuple): a list of sequences
    #
    #     Returns:
    #         LongTensor: x_ori of shape (batch_size, seq_len)
    #         LongTensor: x_aug of shape (batch_size, seq_len)
    #         tuple of List: the cls indices
    #     """
    #     x_ori, x_aug, cls_indices = zip(*batch)
    #     max_len_ori = max([len(x) for x in x_ori])
    #     max_len_aug = max([len(x) for x in x_aug])
    #     maxlen = max(max_len_ori, max_len_aug)
    #     x_ori_new = [xi + [self.tokenizer.pad_token_id]*(maxlen - len(xi)) for xi in x_ori]
    #     x_aug_new = [xi + [self.tokenizer.pad_token_id]*(maxlen - len(xi)) for xi in x_aug]
    #
    #     # decompose the column alignment
    #     cls_ori = []
    #     cls_aug = []
    #     for item in cls_indices:
    #         cls_ori.append([])
    #         cls_aug.append([])
    #
    #         for idx1, idx2 in item:
    #             cls_ori[-1].append(idx1)
    #             cls_aug[-1].append(idx2)
    #
    #     return torch.LongTensor(x_ori_new), torch.LongTensor(x_aug_new), (cls_ori, cls_aug)
class SlopDetector(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int, num_epoch: int, loss_fn, ) -> None:
        super().__init__()




if __name__ == "__main__":



