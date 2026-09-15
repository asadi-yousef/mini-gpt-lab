import torch
from model_v2 import GPT
from data_v2 import get_batch, vocab_size

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(
    vocab_size=vocab_size,
    d_model=320,
    n_heads=8,
    n_layers=6,
    block_size=256,
    dropout=0.1
).to(device)

print(
    "Parameters:",
    f"{sum(p.numel() for p in model.parameters()):,}"
)

x, y = get_batch(
    "train",
    batch_size=8,
    block_size=256
)

x = x.to(device)
y = y.to(device)

logits, loss = model(x, y)

print("logits:", logits.shape)
print("loss:", loss.item())

loss.backward()

print("Forward + backward successful!")