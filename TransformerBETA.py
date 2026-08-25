# - Date: August 25, 2026
# - This is my first implementation of a Transformer from scratch in PyTorch. It uses multi-head self-attention, causal masking, positional embeddings, feed-forward networks, residual connections, and layer normalization. This implementation is a small decoder-style Transformer similar to what you'd see in modern AI systems such as ChatGPT, Claude, and so on. I'll soon be making a new version using PyTorch's built-in Transformer components, while also implementing and experimenting with reinforcement learning.
# - COMPLA

import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
  def __init__(self, embed_size, num_heads):
    super().__init__()

    if embed_size % num_heads != 0:
      raise ValueError("Embed size must be divisible by num heads")

    self.num_heads = num_heads
    self.head_dim = embed_size // num_heads

    self.Wq = nn.Linear(embed_size, embed_size)
    self.Wk = nn.Linear(embed_size, embed_size)
    self.Wv = nn.Linear(embed_size, embed_size)
    self.Wo = nn.Linear(embed_size, embed_size)

  def forward(self, x):
    Q = self.Wq(x)
    K = self.Wk(x)
    V = self.Wv(x)

    Q = Q.reshape(Q.shape[0], Q.shape[1], self.num_heads, self.head_dim)
    K = K.reshape(K.shape[0], K.shape[1], self.num_heads, self.head_dim)
    V = V.reshape(V.shape[0], V.shape[1], self.num_heads, self.head_dim)

    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)


    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(K.shape[-1])

    mask = torch.triu(
        torch.ones(scores.shape[-2:], device=scores.device),
        diagonal=1
    ).bool()

    scores = scores.masked_fill(mask, float("-inf"))


    attention_weights = torch.softmax(scores, dim=-1)

    output = attention_weights @ V
    output = output.transpose(1, 2)
    output = output.reshape(
        output.shape[0],
        output.shape[1],
        self.num_heads * self.head_dim
    )

    output = self.Wo(output)

    

    return output, attention_weights


class TransformerBlock(nn.Module):
  def __init__(self, embed_size, num_heads, ff_dim):
    super().__init__()

    self.attention = SelfAttention(embed_size, num_heads)
    self.norm1 = nn.LayerNorm(embed_size)
    
    self.ff = nn.Sequential(
        nn.Linear(embed_size, ff_dim),
        nn.ReLU(),
        nn.Linear(ff_dim, embed_size)
    )

    self.norm2 = nn.LayerNorm(embed_size)

  def forward(self, x):
    attention_output, weights = self.attention(x)

    x = x + attention_output
    x = self.norm1(x)
    ff_output = self.ff(x)
    x = x + ff_output
    x = self.norm2(x)

    return x, weights

class Transformer(nn.Module):
  def __init__(
      self,
      embed_size,
      num_heads,
      ff_dim,
      vocab_size,
      max_seq_len,
      num_layers
  ):

    super().__init__()

    self.embedding = nn.Embedding(vocab_size, embed_size)
    self.position_embedding = nn.Embedding(max_seq_len, embed_size)
    
    self.blocks = nn.ModuleList([
        TransformerBlock(embed_size, num_heads, ff_dim)
        for _ in range(num_layers)
    ])

    self.norm = nn.LayerNorm(embed_size)
    self.output_layer = nn.Linear(embed_size, vocab_size)

  def forward(self, tokens):
    x = self.embedding(tokens)

    positions = torch.arange(tokens.shape[1], device=tokens.device)
    pos = self.position_embedding(positions)

    x = x + pos

    for block in self.blocks:
      x, weights = block(x)
    
    x = self.norm(x)
    logits = self.output_layer(x)

    return logits, weights


# ------------------------ TESTING SITE IS DOWN HERE



tokens = torch.tensor([
    [0, 1, 2],
    [3, 4, 5]
])

target = torch.tensor([
    [1, 2, 3],
    [4, 5, 0]
])

model = Transformer(
    embed_size = 4,
    num_heads = 2,
    ff_dim = 16,
    vocab_size = 6,
    max_seq_len = 3,
    num_layers = 2
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5000):

  logits, weights = model(tokens)
  loss = loss_fn(logits.reshape(-1, 6), target.reshape(-1))

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

model.eval()

with torch.no_grad():
    logits, weights = model(tokens)

    predictions = logits.argmax(dim=-1)

    print("Predictions:")
    print(predictions)

    print("Targets:")
    print(target)