# Linkedin Slop Language Model

Usage:
WIP

Disclaimer:
Proof of Concept. No actual data has been used from Linkedin. 



Learning Resources Used:

##############################  ADDING MORE AS I KEEP WORKING ON THIS  ##############################

Tokenizer:
- [Andrej Karpathy's MinBpe](https://github.com/karpathy/minbpe)
- [OpenAI's toktoken](https://github.com/openai/tiktoken)
  
Intuition/Math:
- [Nishant Aklecha's From-Scratch Implementation of Llama3](https://github.com/naklecha/llama3-from-scratch)
- [History of Word Embeddings (back when LDA wasn't unc)](https://arxiv.org/pdf/1301.3781)
- [BF16 Tensor Ops ](https://arxiv.org/pdf/1904.06376)
Architecture, Training, Optimization, Etc...:
- [Sebastian Raschka's amazing guide (skimmed it since I didn't want to copy anyone's methodology too closely...)](https://github.com/rasbt/LLMs-from-scratch)
- [Karpathy's Makemore, another classic, kind of expected](https://github.com/karpathy/makemore)
- [The Nuke That Started It All](https://arxiv.org/pdf/1706.03762)
- [KVCaching](https://huggingface.co/blog/not-lain/kv-caching)
- [Decoupled Weight Decay, and Why I'm AdamW's biggest Fan](https://optimization.cbe.cornell.edu/index.php?title=AdamW)
- [Cosine Annealed Learning Rate with Warm Up](https://www.tutorialexample.com/understand-transformers-get_cosine_schedule_with_warmup-with-examples-pytorch-tutorial/)
- [Seeds and Determinism in Distributed Training](https://stackoverflow.com/questions/62097236/how-to-set-random-seed-when-it-is-in-distributed-training-in-pytorch)

Stuff I think is super cool and wish I had time to disseminate, implement and understand more than surface level
- [THE SSM Paper](https://arxiv.org/pdf/2111.00396)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [MoE Layers](https://arxiv.org/pdf/1701.06538)
- 

Generally Good Reads:
- [Where LLMs were in October of 2024](https://arxiv.org/pdf/2307.06435)
