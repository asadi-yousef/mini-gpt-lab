import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from pathlib import Path

from data import get_batch, vocab_size, chars
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
def save_checkpoint(
    model,
    optimizer,
    step,
    train_loss,
    val_loss,
    path
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,

        "config": {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "block_size": block_size,
        },

        # Needed to reconstruct the tokenizer later
        "chars": chars,
    }

    torch.save(checkpoint, path)

def plot_losses(steps, train_losses, val_losses):
    plt.figure(figsize=(8, 5))

    plt.plot(steps, train_losses, label="Train loss")
    plt.plot(steps, val_losses, label="Validation loss")

    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("GPT Training Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_curve.png")
    plt.show()

@torch.no_grad()
def estimate_loss(model):
    model.eval()

    losses = {}

    for split in ["train", "val"]:
        split_losses = []

        for _ in range(eval_iters):
            x, y = get_batch(
                split,
                batch_size=batch_size,
                block_size=block_size
            )

            x, y = x.to(device), y.to(device)

            _, loss = model(x, y)
            split_losses.append(loss.item())

        losses[split] = sum(split_losses) / len(split_losses)

    model.train()

    return losses


def train(model, optimizer):
    steps = []
    train_losses = []
    val_losses = []

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")

    for step in range(max_steps):

        x, y = get_batch(
            "train",
            batch_size=batch_size,
            block_size=block_size
        )

        x, y = x.to(device), y.to(device)

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == max_steps - 1:
            losses = estimate_loss(model)

            train_loss = losses["train"]
            val_loss = losses["val"]

            steps.append(step)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(
                f"step {step:5d} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    train_loss,
                    val_loss,
                    checkpoint_dir / "best.pt"
                )

                print(
                    f"saved new best checkpoint "
                    f"(val loss {val_loss:.4f})"
                )

    # Always save the final state too
    save_checkpoint(
        model,
        optimizer,
        max_steps - 1,
        train_losses[-1],
        val_losses[-1],
        checkpoint_dir / "final.pt"
    )

    plot_losses(
        steps,
        train_losses,
        val_losses
    )

def main():
    print(f"Using device: {device}")

    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        block_size=block_size
    ).to(device)

    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    train(model, optimizer)


if __name__ == "__main__":
    main()