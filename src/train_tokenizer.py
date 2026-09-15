from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


SPECIAL_TOKENS = [
    "<|unk|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]

tokenizer = Tokenizer(
    BPE(unk_token="<|unk|>")
)

tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False
)

tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=ByteLevel.alphabet(),
)

tokenizer.train(
    ["data/dialogue/train.txt"],
    trainer
)

tokenizer.save(
    "data/dialogue/tokenizer.json"
)

print("Vocabulary size:", tokenizer.get_vocab_size())