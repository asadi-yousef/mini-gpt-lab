import torch
from tokenizers import Tokenizer

from model_v2 import GPT


device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Load tokenizer
# -------------------------

tokenizer = Tokenizer.from_file(
    "data/dialogue_v2/tokenizer.json"
)


USER_ID = tokenizer.token_to_id("<|user|>")
ASSISTANT_ID = tokenizer.token_to_id("<|assistant|>")
END_ID = tokenizer.token_to_id("<|end|>")


# -------------------------
# Load best v2 checkpoint
# -------------------------

checkpoint = torch.load(
    "checkpoints/dialogue_v2/best.pt",
    map_location=device,
    weights_only=True
)

config = checkpoint["config"]

model = GPT(**config).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# -------------------------
# Generation
# -------------------------

@torch.no_grad()
def generate_response(
    prompt,
    max_new_tokens=256,
    temperature=0.7,
    top_k=30
):
    ids = tokenizer.encode(prompt).ids

    idx = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device
    )

    response_ids = []

    for _ in range(max_new_tokens):

        # Model can only see block_size tokens
        idx_cond = idx[:, -model.block_size:]

        logits, _ = model(idx_cond)

        # Next-token logits
        logits = logits[:, -1, :]

        # Temperature
        logits = logits / temperature

        # Top-k sampling
        if top_k is not None:
            values, _ = torch.topk(
                logits,
                min(top_k, logits.shape[-1])
            )

            cutoff = values[:, [-1]]

            logits = logits.masked_fill(
                logits < cutoff,
                float("-inf")
            )

        probs = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        token_id = next_token.item()

        # Stop when model ends its answer
        if token_id == END_ID:
            break

        # Also stop if it tries to become the user
        if token_id == USER_ID:
            break

        response_ids.append(token_id)

        idx = torch.cat(
            (idx, next_token),
            dim=1
        )

    return tokenizer.decode(
        response_ids,
        skip_special_tokens=True
    ).strip()


# -------------------------
# Interactive chat
# -------------------------

print(
    f"MiniGPT v2 ready on {device}. "
    "Type 'exit' to quit.\n"
)

while True:

    try:
        user_input = input("You: ")

    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        break

    if user_input.lower() in {"exit", "quit"}:
        break

    if not user_input.strip():
        continue

    prompt = (
        "<|user|>\n"
        + user_input.strip()
        + "\n<|assistant|>\n"
    )

    response = generate_response(prompt)

    print(f"\nMiniGPT v2: {response}\n")
