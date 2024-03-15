import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import sys

#hyperparams
lr = 1e-3
batch_size = 32
block_size = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    sys.exit(0)
max_iters = 801
print_iters = 100
eval_iters = 10
eval_interval = 100
n_embed= 384
n_heads = 4
n_layers = 6
dropout = 0.2
# ---------

with open('sherlock.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Obtain unique characters in dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)

# We will tokenize our vocabulary by building a character level language model. We will represent each
# character as an integer. Sub-word tokenizers are also possible (chat-gpt uses tiktoken)
# We first create a mapping from characters to integers using a dictionary
chtoi = {ch:i for i,ch in enumerate(chars)}
itoch = {i:ch for i,ch in enumerate(chars)}

# Take a string, output list of integers.
encode = lambda s:        [chtoi[ch] for ch in s]
#Take a list of integers, output string.
decode = lambda list_int: ''.join([itoch[i] for i in list_int])

def get_batch(split, train_data, val_data):
    """
    We obtain a context and target tensor of size (batch_size, block_size)
    """
    data = train_data if split=="train" else val_data
    ix = torch.randint(low=0, high=len(data)-block_size, size=(batch_size,))

    # We now turn horizontally
    X = torch.vstack([data[i:i+block_size] for i in ix])
    Y = torch.vstack([data[i+1:i+block_size+1] for i in ix])

    return X.to(device),Y.to(device)

@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            
            # The loss is the mean loss over all the batches, and tokens in each sentence.
            # When we call .backward() on the mean loss, PyTorch computes the gradients for each 
            # individual loss in the batch.
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
  
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        _,T,_ = x.shape   # (B, T, n_embed)
        K = self.key(x)   
        Q = self.query(x) 
        
        # Einsum is easier way to view the batched matrix multiplication. 
        wei = torch.einsum('ijk, ilk -> ijl', Q, K) * self.head_size**(-0.5) # (B, T, n_embed) @ (B, n_embed, T) -> (B, T, T)
        
        # We now mask values so that decoder can't look at future tokens
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        
        # We then incorporate the value matrix
        V = self.value(x) 
        out = wei @ V 
        return out


class MultiHead(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.linear = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.dropout(self.linear(x))
        return x

class FeedForward(nn.Module):
    def __init__(self, n_embed) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
        nn.Linear(n_embed, 4*n_embed),
        nn.ReLU(),
        nn.Linear(4*n_embed, n_embed),
        nn.Dropout(dropout),
        ).to(device)
        
    def forward(self, x):
        return self.ffn(x)

class Block(nn.Module):
  def __init__(self, n_embed, n_heads) -> None:
    super().__init__()
    self.head_size = n_embed // n_heads
    self.sa_head = MultiHead(n_heads, self.head_size)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)
    self.feedforward = FeedForward(n_embed)

  def forward(self, x):
    # We use norm before
    x = x + self.sa_head(self.ln1(x)) 
    x = x + self.feedforward(self.ln2(x)) 

    return x

class LLM(nn.Module):
  def __init__(self,vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
    self.position_embedding_table = nn.Embedding(block_size,n_embed)
    self.lm_head = nn.Linear(n_embed,vocab_size)
    self.blocks = nn.Sequential(*[Block(n_embed,n_heads=n_heads) for _ in range(n_layers)])
    self.ln_f = nn.LayerNorm(n_embed) # final layer norm

  def forward(self, idx, targets=None):
    # idx = idx[:,-block_size:]
    B,T = idx.shape 
    tok_emb = self.token_embedding_table(idx) # (B,T,n_embed)
    pos_emb = self.position_embedding_table(torch.arange(T,device=device)) # (T,n_embed)
    
    x = tok_emb + pos_emb # (B,T,n_embed)
    x = self.blocks(x) # (B,T,n_embed)
    x = self.ln_f(x) # (B,T,n_embed)
    logits = self.lm_head(x) # (B,T,vocab_size)
    
    if targets is None:
      loss = None
    else:
      B,T,V = logits.shape
      logits = logits.view(B*T,V)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
      logits = logits.view(B,T,V)
      
    return logits, loss

  def generate(self, idx, max_new_tokens, temperature=1.0):
    # idx is (B,T)
    idx_next = []
    
    # Autoregressively generate new tokens
    for i in range(max_new_tokens):
      idx_cond = idx[:,-block_size:]
      logits, loss = self(idx_cond)
      last_timestep = logits[:,-1,:] / temperature # (B,V)
      probs = F.softmax(last_timestep, dim=1)
      next_token = torch.multinomial(probs, num_samples=1) # B
      idx = torch.cat((idx, next_token), dim=1)
      
    return idx

if __name__ == "__main__":
    # We now encode entire "sherlock.txt" and save it in a torch tensor.
    data = torch.tensor(encode(text)) 

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Length of dataset: {len(text)} characters.")

    print("vocab_size: ", vocab_size)
    print("Vocabulary: ", "".join(chars))

    model = LLM(vocab_size).to(device)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Number of trainable parameters: {round(num_params,3)} million")

    for iter in tqdm(range(max_iters)):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, eval_iters, train_data, val_data)
            print(f"iter: {iter}  train_loss: {losses['train']:.4f}  val_loss: {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train', train_data, val_data)
        xb.to(device)
        yb.to(device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "saved_model.pth")
    print("Trained model has been saved.")
