from tokenization.build_vocab import pre_tokenize, train_bpe, construct_final_vocab
from tests.consts_test import EXPECTED_MERGE_OUTPUT, EXPECTED_VOCAB

def make_sure_emojis_work():
    assert pre_tokenize("🚀") == ['<BOW>', '🚀', '<EOW>']

def test_metrics():
    assert train_bpe(["I am upskilling fresher 👳️"], n_merges=50) == EXPECTED_MERGE_OUTPUT
    assert construct_final_vocab(EXPECTED_MERGE_OUTPUT, None) == EXPECTED_VOCAB
