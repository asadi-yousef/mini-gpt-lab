import torch
from tokenizers import Tokenizer


tokenizer = Tokenizer.from_file(
    "data/dialogue_v2/tokenizer.json"
)

USER_ID = tokenizer.token_to_id("<|user|>")
ASSISTANT_ID = tokenizer.token_to_id("<|assistant|>")
END_ID = tokenizer.token_to_id("<|end|>")

vocab_size = tokenizer.get_vocab_size()


with open(
    "data/dialogue_v2/train.txt",
    "r",
    encoding="utf-8"
) as f:
    train_text = f.read()

with open(
    "data/dialogue_v2/val.txt",
    "r",
    encoding="utf-8"
) as f:
    val_text = f.read()


train_ids = tokenizer.encode(train_text).ids
val_ids = tokenizer.encode(val_text).ids

def build_labels(token_ids):
    labels = [-100] * len(token_ids)

    inside_assistant = False

    for i, token_id in enumerate(token_ids):

        # Assistant response starts AFTER this token
        if token_id == ASSISTANT_ID:
            inside_assistant = True
            continue

        # User text should not contribute to the loss
        if token_id == USER_ID:
            inside_assistant = False
            continue

        # We DO want the model to learn to generate <|end|>
        if token_id == END_ID:
            if inside_assistant:
                labels[i] = token_id

            inside_assistant = False
            continue

        if inside_assistant:
            labels[i] = token_id

    return labels

train_data = torch.tensor(
    train_ids,
    dtype=torch.long
)

val_data = torch.tensor(
    val_ids,
    dtype=torch.long
)

train_labels = torch.tensor(
    build_labels(train_ids),
    dtype=torch.long
)

val_labels = torch.tensor(
    build_labels(val_ids),
    dtype=torch.long
)


def get_batch(split, batch_size, block_size):
    if split == "train":
        data = train_data
        labels = train_labels
    else:
        data = val_data
        labels = val_labels

    xs = []
    ys = []

    while len(xs) < batch_size:
        i = torch.randint(
            0,
            len(data) - block_size - 1,
            (1,)
        ).item()

        x = data[i:i + block_size]
        y = labels[i + 1:i + block_size + 1]

        # Skip chunks containing no assistant targets
        if (y != -100).any():
            xs.append(x)
            ys.append(y)

    return torch.stack(xs), torch.stack(ys)


if __name__ == "__main__":
    print("Vocabulary size:", vocab_size)
    print("Train tokens:", len(train_data))
    print("Validation tokens:", len(val_data))

    test = tokenizer.encode(
        "<|user|>\nWhat is Python?\n<|assistant|>\n"
    )

    print("Tokens:", test.tokens)
    print("IDs:", test.ids)