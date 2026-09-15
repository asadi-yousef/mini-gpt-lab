from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


dataset = load_dataset("OpenAssistant/oasst1")

output_dir = Path("data/dialogue")
output_dir.mkdir(parents=True, exist_ok=True)


def build_conversations(split):
    messages = dataset[split]

    children = defaultdict(list)

    for message in messages:
        if message["parent_id"] is not None:
            children[message["parent_id"]].append(message)

    roots = [
        message
        for message in messages
        if message["parent_id"] is None
        and message["lang"] == "en"
        and not message["deleted"]
    ]

    conversations = []

    for root in roots:
        conversation = []
        current = root

        while current is not None:

            if current["lang"] != "en" or current["deleted"]:
                break

            conversation.append(current)

            replies = [
                reply
                for reply in children[current["message_id"]]
                if reply["lang"] == "en"
                and not reply["deleted"]
            ]

            if not replies:
                break

            ranked = [
                reply
                for reply in replies
                if reply["rank"] == 0
            ]

            current = ranked[0] if ranked else replies[0]
            
        # Make sure every conversation ends with an assistant response
        while conversation and conversation[-1]["role"] != "assistant":
            conversation.pop()
        # Require at least User -> Assistant
        if (
            len(conversation) >= 2
            and conversation[0]["role"] == "prompter"
            and conversation[1]["role"] == "assistant"
        ):
            conversations.append(conversation)

    return conversations


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
    conversations = build_conversations(split)

    with open(
        output_dir / filename,
        "w",
        encoding="utf-8"
    ) as f:
        for conversation in conversations:
            f.write(format_conversation(conversation))
            f.write("\n\n")

    print(
        f"{split}: saved {len(conversations)} conversations "
        f"to {output_dir / filename}"
    )


save_split("train", "train.txt")
save_split("validation", "val.txt")