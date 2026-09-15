# mini-gpt-lab

A hands-on project for building and training a GPT-style decoder-only Transformer from scratch in PyTorch.

The goal of this project was not to build a production chatbot, but to understand the full language-model pipeline by implementing it directly: tokenization, causal self-attention, multi-head attention, Transformer blocks, next-token training, checkpointing, sampling, dialogue formatting, regularization, and assistant-style fine-tuning.

## What I Built

The model is implemented from scratch with PyTorch and includes:

- Decoder-only Transformer architecture
- Causal self-attention
- Multi-head attention
- Learned token and positional embeddings
- Pre-norm residual Transformer blocks
- GELU feed-forward networks
- AdamW training
- Validation-loss tracking and checkpointing
- Autoregressive text generation
- Temperature and top-k sampling
- Byte-level BPE tokenization
- Weight tying between token embeddings and the language-model head
- Dropout and gradient clipping
- Assistant-only target masking for dialogue training

## Experiments

### 1. Character-Level Shakespeare GPT

The first model used Tiny Shakespeare with a character-level tokenizer.

This experiment was mainly used to implement and verify the Transformer architecture end to end.

The model learned Shakespeare-like formatting, character names, punctuation, and local text patterns, but generated mostly pseudo-English text.

### 2. BPE Dialogue Model — v1

The next experiment switched to the OpenAssistant OASST1 dialogue dataset and a learned byte-level BPE tokenizer.

Configuration:

- Vocabulary size: 8,000
- Context length: 256
- `d_model`: 256
- Attention heads: 8
- Transformer layers: 6
- Approximately 8.9M parameters

The model learned assistant-like surface patterns such as Markdown, code blocks, lists, and phrases like `"Sure! Here is..."`, but semantic coherence remained weak.

### 3. Cleaner Dialogue Model — v2

The final experiment improved both the data pipeline and model architecture.

Data improvements:

- Reconstructed OASST1 conversation paths from message trees
- Filtered to English, non-deleted examples
- Preferred higher-ranked assistant responses
- Ensured examples ended with assistant responses
- Used assistant-only loss while keeping the full user conversation visible as context

Model improvements:

- Vocabulary reduced from 8,000 to 4,000 BPE tokens
- `d_model` increased from 256 to 320
- 8 attention heads
- 6 Transformer layers
- Context length: 256
- Dropout: 0.1
- Tied token embedding and output projection weights
- Gradient clipping
- Approximately 8.75M parameters

The best validation loss reached approximately **3.91**.

The model became noticeably better at reproducing assistant-style responses and conversational structure, but still lacked reliable semantic knowledge and could generate locally plausible but incorrect or repetitive text.

## Key Lesson

The project made the distinction between **language-model pretraining** and **assistant fine-tuning** very clear.

Training directly on assistant responses can teach a model how assistant answers should look, but a useful general-purpose chatbot first needs a strong underlying language model.

A more realistic training pipeline would be:

```text
large general text corpus
        ↓
next-token pretraining on all tokens
        ↓
strong base language model
        ↓
instruction / conversation dataset
        ↓
assistant-focused fine-tuning
```

At this project's scale, the limiting factors became data volume, model capacity, compute, and pretraining quality rather than the basic Transformer implementation.

## Project Structure

```text
mini-gpt-lab/
├── data/
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   ├── data_v2.py
│   ├── model_v2.py
│   ├── train_v2.py
│   ├── generate_v2.py
│   ├── prepare_dialouge.py
│   ├── prepare_dialogue_v2.py
│   ├── train_tokenizer.py
│   └── train_tokenizer_v2.py
├── requirements.txt
├── LICENSE
└── README.md
```

Model checkpoints and generated training artifacts are intentionally excluded from Git.

## Running the Project

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Prepare the dataset and tokenizer, train the desired model version, and run its generation script.

For the dialogue experiments, OASST1 is downloaded through the Hugging Face `datasets` library.

## Conclusion

This project reached its original goal: understanding GPT by building it rather than simply using a pretrained model.

Building a genuinely useful chatbot from scratch would require substantially more general-language pretraining data, a larger model, more compute, and then a separate assistant fine-tuning stage.

## References

- Andrej Karpathy — *Let's build GPT: from scratch, in code, spelled out*
- Vaswani et al. — *Attention Is All You Need*
- Radford et al. — GPT / GPT-2
- OpenAssistant OASST1

## License

MIT License.