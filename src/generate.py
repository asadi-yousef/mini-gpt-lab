import torch
from tokenizers import Tokenizer

from model import GPT


device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Load tokenizer
# -------------------------

tokenizer = Tokenizer.from_file(
    "data/dialogue/tokenizer.json"
)


# -------------------------
# Load model checkpoint
# -------------------------

checkpoint = torch.load(
    "checkpoints/dialogue/best.pt",
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
# Special token IDs
# -------------------------

USER_ID = tokenizer.token_to_id("<|user|>")
ASSISTANT_ID = tokenizer.token_to_id("<|assistant|>")
END_ID = tokenizer.token_to_id("<|end|>")


# -------------------------
# Generate assistant response
# -------------------------

@torch.no_grad()
def generate_response(
    prompt,
    max_new_tokens=200,
    temperature=0.6,
    top_k=20
):
    encoded = tokenizer.encode(prompt).ids

    idx = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device
    )

    response_ids = []

    for _ in range(max_new_tokens):

        # Only feed the most recent context
        idx_cond = idx[:, -model.block_size:]

        logits, _ = model(idx_cond)

        # We only care about prediction after
        # the final token
        logits = logits[:, -1, :]

        # Temperature
        logits = logits / temperature

        # Top-k sampling
        if top_k is not None:
            values, _ = torch.topk(
                logits,
                min(top_k, logits.shape[-1])
            )

            logits[
                logits < values[:, [-1]]
            ] = float("-inf")

        probs = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        token_id = next_token.item()

        # Assistant response is finished
        if token_id in {END_ID, USER_ID}:
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
    f"MiniGPT ready on {device}. "
    "Type 'exit' to quit.\n"
)

history = ""

while True:

    try:
        user_input = input("You: ")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        break

    if user_input.lower() in {
        "exit",
        "quit"
    }:
        break

    if not user_input.strip():
        continue

    prompt = (
        history
        + "<|user|>\n"
        + user_input
        + "\n<|assistant|>\n"
    )

    response = generate_response(prompt)

    print(f"\nMiniGPT: {response}\n")

    # Keep conversation history
    history = (
        prompt
        + response
        + "\n"
    )