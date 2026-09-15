from pathlib import Path

from datasets import load_dataset


dataset = load_dataset("OpenAssistant/oasst1")

output_dir = Path("data/dialogue_v2")
output_dir.mkdir(parents=True, exist_ok=True)


def build_examples(split):
    messages = dataset[split]

    # Fast lookup:
    # message_id -> full message
    by_id = {
        message["message_id"]: message
        for message in messages
    }

    examples = []

    for message in messages:
        # Every example must end with a valid English assistant response
        if message["role"] != "assistant":
            continue

        if message["lang"] != "en":
            continue

        if message["deleted"]:
            continue
        # If this assistant response was ranked against alternatives,
        # only keep the preferred response.
        if message["rank"] is not None and message["rank"] != 0:
            continue

        if message["review_result"] is False:
            continue

        # Walk backwards from this assistant
        # message to the root of the conversation
        conversation = []

        current = message

        while current is not None:

            if current["lang"] != "en" or current["deleted"]:
                conversation = []
                break

            if current["review_result"] is False:
                conversation = []
                break

            if (
                current["role"] == "assistant"
                and current["rank"] is not None
                and current["rank"] != 0
            ):
                conversation = []
                break

            conversation.append(current)

            parent_id = current["parent_id"]

            if parent_id is None:
                break

            current = by_id.get(parent_id)

            if current is None:
                conversation = []
                break

        if not conversation:
            continue

        # We walked backwards, so restore
        # chronological order.
        conversation.reverse()

        # Valid dialogue should begin with user
        # and end with assistant.
        if (
            conversation[0]["role"] != "prompter"
            or conversation[-1]["role"] != "assistant"
        ):
            continue

        examples.append(conversation)

    return examples

def format_conversation(conversation):
    parts = []

    for message in conversation:

        if message["role"] == "prompter":
            role = "<|user|>"
        else:
            role = "<|assistant|>"

        parts.append(
            f"{role}\n{message['text'].strip()}"
        )

    parts.append("<|end|>")

    return "\n".join(parts)

def save_split(split, filename):
    examples = build_examples(split)

    path = output_dir / filename

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for conversation in examples:
            f.write(
                format_conversation(conversation)
            )

            f.write("\n\n")

    print(
        f"{split}: "
        f"{len(examples)} examples saved to {path}"
    )


save_split("train", "train.txt")
save_split("validation", "val.txt")