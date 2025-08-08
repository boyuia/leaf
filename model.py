import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import os

# --- Hyperparameters ---
# These are the settings for our model. You can experiment with these values.
batch_size = 32  # How many sequences to process in parallel
block_size = 8  # Maximum context length for predictions
max_iters = 3000  # Number of training iterations
eval_interval = 300  # How often to evaluate the model
learning_rate = 1e-2  # The learning rate for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
eval_iters = 200  # Number of iterations for evaluation
n_embd = 32  # The dimension of the token embeddings
n_head = 4  # The number of attention heads in the Multi-Head Attention block
n_layer = 4  # The number of Transformer blocks
dropout = 0.0  # Dropout rate for regularization

# --- Data Preparation ---
# To use this code, you need to create a file named 'dataset.jsonl'
# in the same directory as this script. Each line of the file should be a JSON object
# with 'header' and 'formal_statement' keys, like the example you provided.
file_path = 'dataset.jsonl'

# Process the JSONL data from the file.
corpus = ""
try:
    with open(file_path, 'r') as f:
        for line in f:
            data_point = json.loads(line)
            # Combine the 'header' and 'formal_statement' fields.
            # We add a newline character to separate the two parts of the text.
            corpus += data_point['header'] + '\n' + data_point['formal_statement'] + '\n'
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please create it and add your data.")
    exit()
except json.JSONDecodeError:
    print(f"Error: There was a problem parsing a line in '{file_path}'. Make sure each line is a valid JSON object.")
    exit()
except KeyError:
    print(f"Error: A line in '{file_path}' does not have the 'header' or 'formal_statement' keys. Please check your JSONL file format.")
    exit()

# Check if the corpus is empty after loading the file.
if not corpus:
    print(f"Error: The corpus is empty. This could be because '{file_path}' is empty or contains no valid text.")
    exit()

# Here we create a simple character-level tokenizer.
# The vocabulary consists of all unique characters in the text.
chars = sorted(list(set(corpus)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# Fix the bug in the encode function. The loop variable was 's' instead of 'c'.
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Convert the entire text into a PyTorch tensor.
data = torch.tensor(encode(corpus), dtype=torch.long)

# Create a simple train/validation split.
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# --- Helper Functions ---
# This function gets a random batch of data from either the training or validation set.
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Generate random starting indices for each sequence in the batch.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Stack the sequences to create a batch.
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# This function is used to estimate the model's loss on both the train and validation sets.
# It uses torch.no_grad() to make the process more efficient as we're not training.
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # Set the model to evaluation mode.
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # Set the model back to training mode.
    return out

# --- The Self-Attention Mechanism ---
# This is a single attention head.
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # Linear layers to project the input into key, query, and value vectors.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # A buffer to store a lower-triangular matrix, which prevents future tokens from
        # "seeing" past tokens (decoder-style attention).
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # Dropout layer for regularization.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        
        # Compute the affinity scores (weights).
        # (q @ k.transpose(-2, -1)) is matrix multiplication of q and k transpose.
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # Apply the lower-triangular mask to enforce causality.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Apply softmax to get the attention weights.
        wei = F.softmax(wei, dim=-1)
        self.dropout(wei)

        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

# This combines multiple attention heads in parallel.
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create a list of `Head` modules.
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # A final linear layer to project the concatenated output of all heads.
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the output from each head.
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# This is a simple feed-forward network.
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # A simple linear-ReLU-linear stack.
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# This is a single Transformer block, composed of Multi-Head Attention and a Feed-Forward network.
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        # The attention mechanism.
        self.sa = MultiHeadAttention(n_head, head_size)
        # The feed-forward network.
        self.ffwd = FeedFoward(n_embd)
        # Layer normalization layers.
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Apply self-attention with a residual connection and layer normalization.
        x = x + self.sa(self.ln1(x))
        # Apply feed-forward with another residual connection and layer normalization.
        x = x + self.ffwd(self.ln2(x))
        return x

# --- The Main Language Model ---
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A token embedding table: each integer token gets a vector representation.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # A positional embedding table: each position gets a vector representation.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # A sequence of Transformer blocks.
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        # A final layer normalization.
        self.ln_f = nn.LayerNorm(n_embd)
        # A linear layer to project the final embeddings to the vocabulary size.
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Get token embeddings and positional embeddings.
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        # Add them together to get the final embeddings.
        x = tok_emb + pos_emb  # (B, T, C)
        # Pass through the Transformer blocks.
        x = self.blocks(x)
        x = self.ln_f(x)
        # Project to the vocabulary size.
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Reshape for cross-entropy loss calculation.
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # A function to generate text.
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) tensor of indices in the current context.
        for _ in range(max_new_tokens):
            # Crop idx to block_size, as the model has a limited context.
            idx_cond = idx[:, -block_size:]
            # Get predictions.
            logits, loss = self(idx_cond)
            # Focus only on the last time step.
            logits = logits[:, -1, :]
            # Apply softmax to get probabilities.
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution.
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append the new token to the sequence.
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Training and Generation ---
model = LanguageModel()
m = model.to(device)

# Create a PyTorch optimizer.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Main training loop.
for iter in range(max_iters):
    # Every few iterations, evaluate the loss on both splits.
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data.
    xb, yb = get_batch('train')

    # Forward pass: compute loss.
    logits, loss = model(xb, yb)
    # Backward pass: compute gradients.
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # Update the model parameters.
    optimizer.step()

# --- Generate new text from the trained model ---
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text_indices = m.generate(context, max_new_tokens=20)
print("\nGenerated text:")
print(decode(generated_text_indices[0].tolist()))