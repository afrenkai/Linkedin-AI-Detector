from tokenization.build_vocab import BPETokenizer, clean_decoded_output
from tests.tok_toy_scenarios import get_maps
from tests.consts_test import EXPECTED_VOCAB, TEST_ENCODING, EXPECTED_TOKENS, CLEANING_PATTERN
import regex as re

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

