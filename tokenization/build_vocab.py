from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import regex as re
import json
import pickle
from utils.consts import GPT4_REGEX, DEFAULT_SPECIAL_TOKENS

def read_in_slop(file: str) -> List[str]:
    with open(file, "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]
    return corpus


class BPETokenizer:
    def __init__(self, pattern: Optional[str] = None, spec_tok_list: Optional[List[str]] = None):

        """

        @param pattern: defaults to GPT 4 pattern, but I could pass something a little better in if I find util for it later
        @param spec_tok_list: list of special tokens to assign indices to

        """
        self.pattern = GPT4_REGEX if not pattern else pattern
        self.spec_tok_list = spec_tok_list if spec_tok_list else DEFAULT_SPECIAL_TOKENS
        self.merges: List[Tuple[str, ...]] = []
        self.tok_to_id_map: Dict[str, int] = {}
        self.id_to_tok_map:Dict[int, str] = {}
        self.corpus:List[str] = []

    def regex_split(self, text: str, verbose = False) -> List[str]:
        """
        :param text: text to apply regex to
        :param verbose: whether to yap about the regex
        :return: gpt-4 similar list of string tokens, i.e. ['I', ' am', ' upskilling', ' fresher',
        ' 👳️', ' saar', ',', ' I', ' ❤️', ' coding', ' in', ' Python', '3', ' bhai', ' 💻🚀']
        """
        tok =  re.findall(self.pattern, text)
        if verbose:
            print(f'regex_applied to inp: {tok}')
        return tok


    @staticmethod
    def pre_tokenize(tok: str) -> List[str]:
        chars = re.findall(r"\X", tok)
        return ["<BOW>"] + chars + ["<EOW>"]

    def build_init_vocab(self, corp: List[str]) -> Dict[str, int]:
        vocab: Dict[str, int] = {}
        self.corpus = corp
        for line in corp:
            chunk = self.regex_split(line)
            for word in chunk:
                tokenized = tuple(self.pre_tokenize(word))
                word_str = " ".join(tokenized)
                if word_str in vocab:
                    vocab[word_str] += 1
                else:
                    vocab[word_str] = 1
        return vocab

    @staticmethod
    def get_init_vocab_metrics(vocab: Dict[str, int]) -> Dict[str, int]:
        pairs: Dict[str, int] = {}
        for word_str in vocab:
            tok = word_str.split()
            for i in range(len(tok) - 1):
                pair = f"{tok[i]} {tok[i + 1]}"

                if pair in pairs:
                    pairs[pair] += vocab[word_str]
                else:
                    pairs[pair] = vocab[word_str]
        return pairs


    @staticmethod
    def merge(pair: str, vocab: Dict[str, int], verbose:bool = False) -> Dict[str, int]:
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

    def _extract_base_charset(self, corpus: List[str]) -> List[str]:
        charset = set()
        for line in corpus:
            for ch in re.findall(r"\X", line):
                charset.add(ch)
        return sorted(charset)

    def construct_final_vocab(self) -> Dict[str, int]:
        vocab = {}
        idx = 0
        for (a, b) in self.merges:
            tok = a + b
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
        for ch in self._extract_base_charset(self.corpus):
            if ch not in vocab:
                vocab[ch] = idx
                idx += 1
        for line in self.corpus:
            for chunk in self.regex_split(line):
                for tok in self.pre_tokenize(chunk):
                    if tok not in vocab:
                        vocab[tok] = idx
                        idx += 1
        for special in self.spec_tok_list:
            if special not in vocab:
                vocab[special] = idx
                idx += 1
        return vocab


    def train(self, corpus: List[str], n_merges: int = 100, verbose:bool = False) -> None:
        vocab = self.build_init_vocab(corpus)
        self.corpus = corpus
        self.merges = []
        if verbose:
            print(self.get_init_vocab_metrics(vocab))
        for i in tqdm(range(n_merges)):
            info = self.get_init_vocab_metrics(vocab)
            if not info:
                if verbose:
                    print("empty vocab")
                break
            best_pair = max(info, key=lambda x: info[x])
            self.merges.append(tuple(best_pair.split()))
            vocab = self.merge(best_pair, vocab)
            if verbose:
                print(f'[{i+1}] Merged {best_pair}')
        self.tok_to_id_map = self.construct_final_vocab()
        for tok, idx in self.tok_to_id_map.items():
            self.id_to_tok_map[idx] = tok


    def apply_merge(self, tokens: List[str]) -> List[str]:
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == merge[0] and tokens[i + 1] == merge[1]:
                    tokens = tokens[:i] + [merge[0] + merge[1]] + tokens[i + 2:]
                else:
                    i += 1
        return tokens
    def show_current_map(self):
        if not self.tok_to_id_map:
            raise RuntimeError("Untrained, meaning there are no idxes")
        print(self.tok_to_id_map.keys())

    def encode(self, words)-> List[int]:
        idxs: List[int] = []
        regex_tokens = self.regex_split(words)
        for token in regex_tokens:
            pre_tokens = self.pre_tokenize(token)
            merged_tokens = self.apply_merge(pre_tokens)
            for tok in merged_tokens:
                idxs.append(self.tok_to_id_map.get(tok, self.tok_to_id_map["<UNK>"]))

        return idxs
    #
    def decode(self, ids: List[int]) -> str:
        if not self.id_to_tok_map:
            raise RuntimeError("Tokenizer not yet trained.")
        skip = {"<BOW>", "<EOW>", "<CLS>", "<PAD>"}
        tokens = [self.id_to_tok_map.get(i, "") for i in ids]
        return "".join(t for t in tokens if t and t not in skip)

    def save(self, filepath: str, format: str = "json") -> None:
        """
        Save the tokenizer to json
        
        @param filepath: Path to save the tokenizer
        @param format: Format to save in ("json" or "pickle")
        """
        if not self.tok_to_id_map:
            raise RuntimeError("Tokenizer not yet trained.")
        
        data = {
            "pattern": self.pattern,
            "spec_tok_list": self.spec_tok_list,
            "merges": self.merges,
            "tok_to_id_map": self.tok_to_id_map,
            "id_to_tok_map": self.id_to_tok_map
        }
        
        if format.lower() == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2) # i think this is how you pretty print i dont remember to be honest
        elif format.lower() == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
        else:
            raise ValueError("Format must be 'json' or 'pickle'")

    @classmethod
    def load(cls, filepath: str, format: str = "json") -> "BPETokenizer":
        """
        Loads a tokenizer from a file.
        
        @param filepath: Path to load the tokenizer from
        @param format: Format to load from ("json" or "pickle")
        @return: Loaded BPETokenizer instance
        """
        if format.lower() == "json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif format.lower() == "pickle":
            with open(filepath, "rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError("Format must be 'json' or 'pickle'")
        
        tokenizer = cls(pattern=data["pattern"], spec_tok_list=data["spec_tok_list"])
        tokenizer.merges = data["merges"]
        tokenizer.tok_to_id_map = data["tok_to_id_map"]
        tokenizer.id_to_tok_map = data["id_to_tok_map"]
        
        return tokenizer




def clean_decoded_output(output: str):
    pat = re.compile(r'(<[\w]{3}>)')
    new_str =  re.sub(pat, "", output)
    return new_str




    
