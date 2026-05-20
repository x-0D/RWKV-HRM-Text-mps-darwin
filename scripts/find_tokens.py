from tokenizers import Tokenizer
tok = Tokenizer.from_file('../data_io/trained_tokenizers/bpe/tokenizer.json')
vocab = tok.get_vocab()

# Find special tokens by looking for tokens with special names
special_tokens = [t for t in vocab.keys() if t.startswith('<') or t.startswith(chr(65530))]
print(f"Found {len(special_tokens)} special tokens:")
for t in sorted(special_tokens)[:50]:
    print(f"  {repr(t)}: {vocab[t]}")

# Also look for tokens around the high Unicode range
high_tokens = [t for t in vocab.keys() if any(ord(c) > 65000 for c in t)]
print(f"\nFound {len(high_tokens)} high-unicode tokens:")
for t in sorted(high_tokens)[:20]:
    print(f"  {repr(t)}: {vocab[t]}")
