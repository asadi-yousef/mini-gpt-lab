import torch
from tokenizers import Tokenizer


tokenizer = Tokenizer.from_file(
    "data/dialogue/tokenizer.json"
)

vocab_size = tokenizer.get_vocab_size()


with open(
    "data/dialogue/train.txt",
    "r",
    encoding="utf-8"
) as f:
    train_text = f.read()

with open(
    "data/dialogue/val.txt",
    "r",
    encoding="utf-8"
) as f:
    val_text = f.read()


train_ids = tokenizer.encode(train_text).ids
val_ids = tokenizer.encode(val_text).ids

train_data = torch.tensor(
    train_ids,
    dtype=torch.long
)

val_data = torch.tensor(
    val_ids,
    dtype=torch.long
)


def get_batch(split, batch_size, block_size):
    data = train_data if split == "train" else val_data

    ix = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x, y