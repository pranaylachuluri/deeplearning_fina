# Character-level GPT from Scratch

A small decoder-only transformer (GPT) written from scratch in PyTorch and trained on Tiny Shakespeare. It learns to predict the next character and then writes new Shakespeare-style text.

Built for CSE 676 Deep Learning at the University at Buffalo.

## What's implemented

- **Data pipeline:** downloads Tiny Shakespeare (~1.1M characters), builds a character vocabulary, makes a 90/10 train/validation split and serves 128-character context windows
- **Causal self-attention:** multi-head attention with a lower-triangular mask, so each position only sees earlier characters
- **GPT block:** pre-LayerNorm attention and MLP sublayers with residual connections and dropout
- **GPT model:** token and position embeddings, a stack of blocks, a final LayerNorm and a language-model head
- **Training:** AdamW with cross-entropy loss, validation loss each epoch and checkpoints at epochs 1, 3 and 5
- **Generation:** autoregressive sampling with temperature, cropping the context to the last 128 characters

## Baseline configuration

| Setting | Value |
| --- | --- |
| Context length | 128 characters |
| Embedding size | 64 |
| Attention heads | 4 |
| Layers | 4 |
| MLP hidden size | 128 |
| Dropout | 0.1 |
| Optimizer | AdamW, learning rate 3e-4, batch 64 |
| Training | 5 epochs × 200 steps |

## Run it

```bash
pip install -r requirements.txt
python gpt_template.py
```

This trains the model, writes `training_log.json` and checkpoints to `checkpoints/`, and prints generated text samples.

## Files

| Path | Contents |
| --- | --- |
| `gpt_template.py` | Dataset, model, training loop and text generation |
| `data/input.txt` | Tiny Shakespeare corpus |
| `assignment.pdf` | Assignment handout |
