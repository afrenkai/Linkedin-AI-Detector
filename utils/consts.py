# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py

GPT4_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
DEFAULT_SPECIAL_TOKENS = ["<BOW>", "<EOW>", "<PAD>", "<UNK>", "<CLS>"]