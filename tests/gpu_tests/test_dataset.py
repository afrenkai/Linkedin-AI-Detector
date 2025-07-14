import pytest
from tests.tok_toy_scenarios import EXPECTED_X, EXPECTED_Y
from tokenization.build_vocab import BPETokenizer, read_in_slop
from utils.consts import DEFAULT_BPE_MERGES
from dataset.dataset import create_dl
from utils.determinism import set_seeds

set_seeds(512)

@pytest.fixture(scope="session")
# DISCLOSURE: TEXT GENERATED WITH GPT o3
def sample_texts():
    return [
        "The artificial intelligence revolution has transformed how we approach problem solving in modern computing.",
        "Machine learning algorithms can process vast amounts of data to identify patterns that humans might miss.",
        "Deep learning models, particularly transformers, have achieved remarkable success in natural language processing tasks.",
        "Neural networks with attention mechanisms can focus on relevant parts of input sequences during processing.",
        "The development of large language models has opened new possibilities for human-computer interaction."
    ]

@pytest.fixture(scope="session")
def corpus_file(tmp_path_factory, sample_texts):
    path = tmp_path_factory.getbasetemp() / "test_corpus_simple.txt"
    path.write_text("\n".join(sample_texts), encoding="utf-8")
    return path

@pytest.fixture(scope="session")
def tokenizer(corpus_file):
    corpus = read_in_slop(corpus_file)        
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, n_merges=DEFAULT_BPE_MERGES, verbose=False)
    return tokenizer


def _expected_xy(first_sentence, tokenizer, block_size): 
    ids = tokenizer.encode(first_sentence)
    ids = ids[:block_size + 1] + [tokenizer.tok_to_id_map["<PAD>"]] * max(0, block_size + 1 - len(ids))
    x = ids[:-1]           
    y = ids[1:]           
    return x, y


@pytest.mark.parametrize(
    ("block_size", "batch_size"),
    [(32, 2), (64, 4)],            
)

def test_dataloader_shapes(corpus_file, tokenizer, block_size, batch_size):
    dl = create_dl(corpus_file, block_size, batch_size, tokenizer, split="train")
    x, y = next(iter(dl))
    assert x.shape == y.shape
    assert x.shape == (batch_size, block_size)

def test_expected_first_example(sample_texts, corpus_file, tokenizer):
    block_size, _ = 32, 2
    expected_x, expected_y = _expected_xy(sample_texts[0], tokenizer, block_size)
    assert expected_x[:10] == EXPECTED_X
    assert expected_y[:10] == EXPECTED_Y
