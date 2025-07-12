from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import regex


def read_in_slop(file: str) -> List[str]:
    with open(file, "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]
    return corpus



def pre_tokenize(word: str) -> List[str]:
    chars = regex.findall(r"\X", word)
    return ["<BOW>"] + chars + ["<EOW>"]



def build_init_vocab(corp: List[str]) -> Dict[str, int]:
    vocab = {}
    for line in corp:
        words = line.strip().split()
        for word in words:
            tokenized = tuple(pre_tokenize(word))
            word_str = " ".join(tokenized)
            if word_str in vocab:
                vocab[word_str] += 1
            else:
                vocab[word_str] = 1
    return vocab


def get_init_vocab_metrics(vocab: Dict[str, int]) -> Dict[str, int]:
    pairs = {}
    for word_str in vocab:
        tok = word_str.split()
        for i in range(len(tok) - 1):
            pair = f"{tok[i]} {tok[i + 1]}"

            if pair in pairs:
                pairs[pair] += vocab[word_str]
            else:
                pairs[pair] = vocab[word_str]
    return pairs


def merge(pair: str, vocab: Dict[str, int], verbose = False) -> Dict[str, int]:
    new_voc = {}
    if verbose:
        print(pair)
    a, b = pair.split()
    replacement = a + b
    target = f"{a} {b}"
    for word_str, freq in vocab.items():
        new_word = word_str.replace(target, replacement)
        new_voc[new_word] = freq

    return new_voc

def train_bpe(corpus: List[str], n_merges: int = 100, verbose = False) -> List[Tuple[str, ...]]:
    vocab = build_init_vocab(corpus)
    merges = []
    # print(get_init_vocab_metrics(voc))
    for i in tqdm(range(n_merges)):
        info = get_init_vocab_metrics(vocab)
        if not info:
            print("empty vocab")
            break
        best_pair = max(info, key = info.get)
        merges.append(tuple(best_pair.split()))
        vocab = merge(best_pair, vocab)
        if verbose:
            print(f'[{i+1}] Merged {best_pair}')
    return merges

def encode_word(word: str, merges: List[Tuple[str, ...]]) -> List[str]:
    tok = pre_tokenize(word)
    for merge in merges:
        i = 0
        while i < len(tok) - 1:
            if tok[i] == merge[0] and tok[i + 1] == merge[1]:
                tok = tok[:i] + [merge[0] + merge[1]] + tok[i+2:]
            else:
                i +=1
    return tok


def add_special_chars(vocab: Dict[str, int], spec_chars_list: List[str], idx: int) -> Dict[str, int]:
    for ch in spec_chars_list:
        if ch not in vocab:
            vocab[ch] = idx
            idx += 1
    return vocab

def construct_final_vocab(merges: List[Tuple[str, ...]], spec_tok_list: Optional[List[str]]) -> Dict[str, int]:
    vocab = {}
    idx = 0
    for merge in merges:
        tok = merge[0] + merge[1]
        if tok not in vocab:
            vocab[tok] = idx
            idx += 1
    for ch in list("abcdefghijklmnopqrstuvwxyz"):
        if ch not in vocab:
            vocab[ch] = idx
            idx += 1
    if spec_tok_list:
        vocab = add_special_chars(vocab, spec_tok_list, idx = idx)

    return vocab



if __name__ == "__main__":
    # print(pre_tokenize("🚀"))
    corpus = ["I am upskilling fresher 👳️"]
    #
    #
    merges = train_bpe(corpus, n_merges=50)
    # spec_tok_list = ['<BOS>', '<CLS>', '<PAD>', '<UNK>', '<EOS>', '<BOW>', '<BOW>', '<EOW>']
    #
    final_vocab = construct_final_vocab(merges, spec_tok_list = None)
    print(final_vocab)

