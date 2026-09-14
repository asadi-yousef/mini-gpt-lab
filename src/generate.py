import torch

from model import GPT


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(
    "checkpoints/best.pt",
    map_location=device,
    weights_only=True
)

config = checkpoint["config"]
chars = checkpoint["chars"]

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(text):
    return [stoi[ch] for ch in text]

def decode(tokens):
    return "".join(itos[i] for i in tokens)


model = GPT(**config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"MiniGPT ready on {device}. Type 'exit' to quit.\n")

while True:
    try:
        prompt = input("You: ")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        break

    if prompt.lower() in {"exit", "quit"}:
        break

    if not prompt:
        continue

    try:
        encoded = encode(prompt)
    except KeyError as e:
        print(f"Unknown character: {e}")
        continue

    idx = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device
    )

    prompt_length = idx.shape[1]

    generated = model.generate(
        idx,
        max_new_tokens=300
    )

    response_ids = generated[0, prompt_length:].tolist()
    response = decode(response_ids)

    print(f"\nGPT: {response}\n")