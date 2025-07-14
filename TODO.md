- gradient ckpt
- early stopping
- param validation (config could just be insane since I took it from jamba)
- cuda oom handling
- tensorboard/wandb viz and logging
- ckpt on val loss
- DL num_workers should adapt to gpus (currently hardcoded to 0)
- amp is being weird, no idea if it's LSP or something else
- docs, typehints
- magic numbers galore, not readable. was wonderful when I needed to test the math, not sustanable and a lot of debt (pirate software moment)
- roundtrip w/o clean fn

FAILED tests/gpu_tests/test_model.py::test_logits - RuntimeError: The size of tensor a (16) must match the size of tensor b (1024) at non-singleton dimension 1
FAILED tests/gpu_tests/test_model.py::test_expected_first_example - assert [tensor(6), t...nsor(38), ...] == [37, 13, 21, 2, 37, 6, ...]