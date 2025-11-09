import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

Tokenizer = Tokenizer(BPE(unk_token="[UNK]")) # Initialize a BPE Tokenizer with an unknown token
Tokenizer.pre_tokenizer = Whitespace() # Use whitespace pre-tokenization

trainer = BpeTrainer(vocab_size=5000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]) # Define a BPE trainer with special tokens

files = ["input.txt"] # Get the training data file path from user input

print("Training tokenizer...")
Tokenizer.train(files, trainer) # Train the tokenizer on the provided files

Tokenizer .save("bpe_tokenizer.json") # Save the trained tokenizer to a file

print("Tokenizer trained and saved as 'bpe_tokenizer.json'")# Save the trained tokenizer to a file