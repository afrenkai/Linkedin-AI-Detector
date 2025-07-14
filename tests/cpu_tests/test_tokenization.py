from tokenization.build_vocab import BPETokenizer
from tests.tok_toy_scenarios import get_maps
from tests.consts_test import EXPECTED_IDT_MAP, TEST_ENCODING, EXPECTED_TID_MAP,  TOKENIZED_CORP, CORP_IDX_TEST
from utils.consts import TID_MAP_TUPLE_LOC, IDT_MAP_TUPLE_LOC

tokenizer  = BPETokenizer()
corpus = ["I am upskilling fresher 👳️"]


def make_sure_emojis_work():
    assert tokenizer.pre_tokenize(corpus[CORP_IDX_TEST]) == TOKENIZED_CORP

def test_tok_to_id():
    assert get_maps(tokenizer, corpus)[TID_MAP_TUPLE_LOC] == EXPECTED_TID_MAP
def test_id_to_tok():
    assert get_maps(tokenizer, corpus)[IDT_MAP_TUPLE_LOC] == EXPECTED_IDT_MAP
    
def test_map_equality():
    assert len(get_maps(tokenizer, corpus)[TID_MAP_TUPLE_LOC]) == len(get_maps(tokenizer, corpus)[IDT_MAP_TUPLE_LOC])

def test_roundabout():
    assert tokenizer.encode(corpus[CORP_IDX_TEST]) == TEST_ENCODING
    assert tokenizer.decode(TEST_ENCODING) == corpus[CORP_IDX_TEST]

