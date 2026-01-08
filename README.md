# GPT‑2 Architecture — From Scratch

A compact, educational implementation of the GPT‑2 architecture in PyTorch. The repository reimplements core pieces of a transformer language model (token & position embeddings, transformer blocks with multi‑head attention, layer normalization, and LM head) so you can study, train, and generate text from a small GPT‑style model. 

## Goals

- Provide a readable-from-scratch GPT‑2 implementation for learning and experimentation.
- Include minimal training and generation utilities so you can train on small datasets and sample text.
- Keep dependencies small so the project runs on a single machine or in Colab.

## Repository layout

- `gpt2/` — model entrypoint and high‑level API (`gpt_model.py`).
- `gpt2_config/` — model configuration (`config.py`).
- `transformer_block/` — transformer block implementation (`transformer_block.py`).
- `attention/` — multi‑head attention implementation.
- `layer_norm/` — custom layer normalization.
- `mlp/` — feed‑forward network used inside transformer blocks.
- `training/` — training and evaluation helpers (`trainer.py`).
- `data/` and `final_dataset/` — dataset helpers and example dataset shards for training.
- `generation/` — simple autoregressive generation helper (`generate_text.py`).
- `sentencePiece_tokenizer/` — a SentencePiece model and vocab trained using Nepali Dataset (planning to use for Nepali Dataset).

## Dependencies

Core requirements are listed in `requirements.txt`. At minimum you need:

- Python 3.8+ (project tested with 3.11)
- PyTorch
- sentencepiece (if you want to use the included tokenizer model)
- tiktoken

Install dependencies (example):

```bash
pip install -r requirements.txt
pip install torch          # install the correct torch build for your platform/GPU
pip install sentencepiece
```

## Quick start — training

The code provides a minimal training loop in `training/trainer.py`. Typical steps:

1. Prepare a dataset (see `data/dataset.py` and the `final_dataset/` shards in the repo).
2. Create a `GPT2Config` from `gpt2_config/config.py` and instantiate `gpt2.gpt_model.GPT2(cfg)`.
3. Build PyTorch DataLoaders that return `(input_batch, target_batch)` where targets are input shifted by one token.
4. Call `training.trainer.train_model(...)` with model, loaders, optimizer and training params.

The trainer exposes these helper functions you can call from a script or notebook:

- `train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer)`
- `evaluate_model(model, train_loader, val_loader, device, eval_iter)`

Example:

```python
from gpt2_config.config import GPT2Config
from gpt2.gpt_model import GPT2
from training.trainer import train_model

cfg = GPT2Config()
model = GPT2(cfg).to(device)
# build optimizer and dataloaders...
train_model(model, train_loader, val_loader, optimizer, device, num_epochs=3, eval_freq=200, eval_iter=10, start_context=None, tokenizer=None)
```

## Quick start — generation

Use `generation/generate_text.py` to autoregressively generate tokens from a starting context. The function `generate_text(model, idx, max_new_tokens, context_size)` expects:

- `model`: the GPT2 model instance
- `idx`: torch.LongTensor of shape `(B, T)` containing token ids for the prompt
- `max_new_tokens`: number of tokens to sample
- `context_size`: maximum context length (model context)

The helper currently uses greedy decoding (argmax). Replace sampling logic with top‑k/top‑p or temperature sampling for more diverse output.

## Configuration

Key configuration parameters live in `gpt2_config/config.py` (embedding dim, number of layers, heads, vocab size, context length, dropout). Customize them for smaller or larger experiments. The default values mirror GPT‑2 small dimensions.

## Tests & Notebooks

- Jupyter notebooks in the repo (e.g., `gpt2/generate_text.ipynb`, `attention/mha.ipynb`, `training/train_colab.ipynb`) walk through components and include runnable examples.



