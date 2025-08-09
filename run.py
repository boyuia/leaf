import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import os

# --- Model Hyperparameters ---
# These must match the hyperparameters used during training.
batch_size = 32
block_size = 8
n_embd = 32
n_head = 4
n_layer = 4
dropout = 0.0

# --- Model Definition ---
# The model architecture must be defined here so that `torch.load` knows
# how to reconstruct the model from the saved state dictionary.
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
        self.vocab_size = vocab_size
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Utility functions and model loading ---
# Read the dataset to create the vocabulary, which must match the model's vocabulary
file_path = 'dataset.jsonl'
corpus = ""
try:
    with open(file_path, 'r') as f:
        for line in f:
            data_point = json.loads(line)
            corpus += data_point['header'] + '\n' + data_point['formal_statement'] + '\n'
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Cannot create vocabulary.")
    exit()
except (json.JSONDecodeError, KeyError):
    print(f"Error: There was a problem parsing a line in '{file_path}'.")
    exit()

if not corpus:
    print("Error: The corpus is empty.")
    exit()

chars = sorted(list(set(corpus)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the trained model
model = LanguageModel(vocab_size, block_size, n_embd, n_head, n_layer, dropout)
model_file = 'model.pt'
if not os.path.exists(model_file):
    print(f"Error: The model file '{model_file}' was not found.")
    exit()

try:
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()  # Set the model to evaluation mode
    model.to(device)
    print("Model loaded successfully. Type a message to chat with the model (or 'exit' to quit).")
except RuntimeError as e:
    print(f"Error: A problem occurred while loading the model state dictionary: {e}")
    print("This may be due to a vocabulary mismatch. Please ensure you have a `model.pt` file trained with the current `dataset.jsonl`.")
    exit()

# --- Terminal Chat Loop ---
while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Encode the user's input
        context = torch.tensor(encode(user_input), dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate new text
        generated_text_indices = model.generate(context, max_new_tokens=50)
        generated_text = decode(generated_text_indices[0].tolist())

        # Print the model's response
        print(f"Model: {generated_text[len(user_input):]}")
    except Exception as e:
        print(f"An error occurred: {e}")
        continue