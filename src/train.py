import torch
from torch.optim import AdamW

from data import get_batch, vocab_size
from model import GPT

# ---------------------------
# Hyperparameters
# ---------------------------

# Data
batch_size = 32
block_size = 128

# Model
d_model = 128
n_heads = 4
n_layers = 4

# Training
learning_rate = 3e-4
weight_decay = 0.01
max_steps = 5000

# Evaluation
eval_interval = 250
eval_iters = 100

# Reproducibility
seed = 42

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(seed)