import random
import numpy as np
from tokenization.build_vocab import BPETokenizer, clean_decoded_output, read_in_slop
from tests.tok_toy_scenarios import get_maps
from tests.consts_test import EXPECTED_VOCAB, TEST_ENCODING, EXPECTED_TOKENS, CLEANING_PATTERN
import regex as re
from tokenization.build_vocab import BPETokenizer
from utils.consts import DEFAULT_BPE_MERGES
from dataset.dataset import create_dl
import os

def make_sure_emojis_work():
    assert BPETokenizer.pre_tokenize("🚀") == ['<BOW>', '🚀', '<EOW>']

def test_tokenizer():
    tokenizer = BPETokenizer()
    corpus = ["I am upskilling fresher 👳️"]
    assert (get_maps(tokenizer, corpus)[0] == EXPECTED_TOKENS)
    assert (get_maps(tokenizer, corpus)[1] == EXPECTED_VOCAB)
    assert len(get_maps(tokenizer, corpus)[0]) == len(get_maps(tokenizer, corpus)[1])
    assert (tokenizer.encode(str(corpus)) == TEST_ENCODING)
    assert clean_decoded_output(tokenizer.decode(TEST_ENCODING)) == re.sub(CLEANING_PATTERN, "", str(corpus))

