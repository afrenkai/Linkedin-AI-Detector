from tokenization.build_vocab import BPETokenizer
from typing import List, Tuple

def get_maps(tokenizer: BPETokenizer, corpus: List[str]) -> Tuple[dict[str, int], dict[int, str]]:
    tokenizer.train(corpus)
    return tokenizer.tok_to_id_map, tokenizer.id_to_tok_map


EXPECTED_X = [37, 13, 21, 2, 37, 6, 4, 32, 22, 19]
EXPECTED_Y = [13, 21, 2, 37, 6, 4, 32, 22, 19, 22]
