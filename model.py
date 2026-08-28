# If you're wondering about the new ai looking comments its just cause I thought the headers looked cool, although I did have AI help me study this, but this is all human written.
# - COMPLA
# August 27, 2026

#========================================================================================
#                                    INITIALIZATION
#========================================================================================
import time
import requests
import torch.nn.functional as F
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
  def __init__(self, embed_size, num_heads, num_kv_heads, ff_dim, max_seq_len):
    super().__init__()

    self.attention = Attention(embed_size, num_heads, num_kv_heads, max_seq_len)
    self.norm1 = nn.RMSNorm(embed_size)

    self.ff_gate = nn.Linear(embed_size, ff_dim)
    self.ff_up =   nn.Linear(embed_size, ff_dim)
    self.ff_down = nn.Linear(ff_dim, embed_size)

    self.norm2 = nn.RMSNorm(embed_size)

  def forward(self, x):
    norm_x = self.norm1(x)

    attn_out = self.attention(norm_x)

    x = x + attn_out
    x = self.norm2(x)

    gate = F.silu(self.ff_gate(x))
    up = self.ff_up(x)


    ff_out = gate * up
    ff_out = self.ff_down(ff_out)

    x = x + ff_out

    return x


class RoPE(nn.Module):
  def __init__(self, head_dim, max_seq_len):
    super().__init__()

    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))

    positions = torch.arange(max_seq_len).float()

    freqs = torch.outer(positions, inv_freq)

    self.register_buffer('cos', freqs.cos())
    self.register_buffer('sin', freqs.sin())

  def forward(self, q, k):
    seq_len = q.shape[-2]

    cos = self.cos[:seq_len]
    sin = self.sin[:seq_len]

    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    q = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
    k = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)

    return q, k

class Attention(nn.Module):
  def __init__(self, embed_size, num_heads, num_kv_heads, max_seq_len):
    super().__init__()

    assert embed_size % num_heads == 0
    assert num_heads % num_kv_heads == 0

    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = embed_size // num_heads

    self.q_proj = nn.Linear(embed_size, num_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(embed_size, num_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(embed_size, num_kv_heads * self.head_dim, bias=False)
    self.out_proj = nn.Linear(embed_size, embed_size, bias=False)

    self.rope = RoPE(self.head_dim, max_seq_len)

  def forward(self, x):
    batch_size, seq_len, _ = x.shape

    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
    k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1,2)
    v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1,2)

    q, k = self.rope(q, k)

    groups = self.num_heads // self.num_kv_heads

    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    out = out.transpose(1,2).contiguous()

    out = out.view(batch_size, seq_len, -1)

    return self.out_proj(out)

#========================================================================================
#                                        TRANSFORMER
#========================================================================================

class Transformer(nn.Module):
  def __init__(
      self,
      embed_size,
      num_heads,
      num_kv_heads,
      ff_dim,
      vocab_size,
      max_seq_len,
      num_layers
  ):
      super().__init__()
      self.token_embedding = nn.Embedding(vocab_size, embed_size)
      self.blocks = nn.ModuleList([
          TransformerBlock(embed_size, num_heads, num_kv_heads, ff_dim, max_seq_len) for _ in range(num_layers)
      ])

      self.norm = nn.RMSNorm(embed_size)
      self.output_layer = nn.Linear(embed_size, vocab_size, bias=False)

      self.output_layer.weight = self.token_embedding.weight


  def forward(self, tokens):

    x = self.token_embedding(tokens)

    for block in self.blocks:
      x = block(x)

    x = self.norm(x)
    logits = self.output_layer(x)

    return logits

#========================================================================================
#                                          TRAINING
#========================================================================================


def train_model(epochs, epoch_check):
  model.train()

  torch.cuda.synchronize()
  start_time = time.perf_counter()

  for epoch in range(epochs):
    logits = model(inputs)
    loss = loss_fn(logits.reshape(-1,len(vocab)), targets.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    if epoch % epoch_check == 0 or epoch == epochs - 1:
      print(f"Epoch: {epoch} | Loss: {loss.item():.4f} | Time: {elapsed:2f}s")


  model.eval()

  with torch.no_grad():
      logits = model(inputs)

      predictions = logits.argmax(dim=-1)

#========================================================================================
#                                      TRAINING DATA
#========================================================================================

url = "https://docs.google.com/document/d/17LEafirLbHHlUwi-rwo-Clq1DqDAni0PG_6qgr7w2SA/export?format=txt"
text = requests.get(url).content.decode("utf-8-sig").lower()

words = text.split()

words.append("<unk>")

vocab = sorted(set(words))

word_to_id = {word: id for id, word in enumerate(vocab)}
id_to_word = {id: word for word, id in word_to_id.items()}

tokens = [word_to_id[word] for word in words]

seq_len = 32
max_seq_len = 32

inputs = []
targets = []

for i in range(len(tokens) - seq_len):
  inputs.append(tokens[i:i+seq_len])
  targets.append(tokens[i+1:i+seq_len+1])

inputs = torch.tensor(inputs)
targets = torch.tensor(targets)

#========================================================================================
#                                      TEXT GENERATION
#========================================================================================

def generate(input_text, num_words):
  words = input_text.lower().split()
  tokens = [word_to_id["<bos>"], word_to_id["<user>"]]

  for word in words:
    tokens.append(word_to_id.get(word, word_to_id["<unk>"]))

  tokens.append(word_to_id["<assistant>"])

  response_start = len(tokens)

  for _ in range(num_words):
    input_tokens = torch.tensor(
        [tokens[-seq_len:]],
        device=device
        )

    with torch.no_grad():
      logits = model(input_tokens)
      next_token = logits[:, -1, :].argmax(dim=-1).item()

      tokens.append(next_token)

      if next_token == word_to_id["<eos>"]:
        break

  response_tokens = tokens[response_start:]

  generated_words = [
      id_to_word[token]
      for token in response_tokens
      if token not in [
        word_to_id["<bos>"],
        word_to_id["<user>"],
        word_to_id["<assistant>"],
        word_to_id["<eos>"]
      ]
    ]

  return generated_words

#========================================================================================
#                                        MODEL DATA
#========================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    embed_size=32,
    num_heads=4,
    num_kv_heads=2,
    ff_dim=64,
    vocab_size=len(vocab),
    max_seq_len=max_seq_len,
    num_layers=2
)

model = model.to(device)
model = torch.compile(model)
inputs = inputs.to(device)
targets = targets.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()


#========================================================================================
#                                      RESPONSE SYSTEM
#========================================================================================

running = True
trained = False

while running:
  command = input("\nCommand:").strip().lower()

  if command == "/train":
    train_model(500,50)
    trained = True

  elif command == "/chat":
    if trained:
      print("Chat started, use /end to end program.")
      while running:
        user_prompt = input("User: ")

        if user_prompt == "/end":
          running = False
          break

        words = generate(user_prompt, 50)
        ai_prompt = ' '.join(words)
        print()
        print(f"USER Question: {user_prompt}")
        print(f"AI Response: {ai_prompt}")

    else: 
      print("You need to train the AI first!")

  else:
    print("Invalid Command.")

