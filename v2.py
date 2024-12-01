import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # independent sequences that will be processed in parallel
block_size = 8 # maximum context length
max_iters = 5000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-3
n_embd = 32
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# encoding/decoding functions
stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # takes a string, returns list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # takes list of integers, returns a string

# train/val splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    """
    returns a batch (x, y) of batch_size = 4 arrays
    arrays are of size block_size = 8
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # get batch_size=4 random numbers from len(data) - block_size. minus block_size is to prevent overflow
    x = torch.stack([data[i:i+block_size] for i in ix]) # stack batch_size=4 arrays cotaining block_size=8 elements from data starting from index in ix
    y = torch.stack([data[i+1:i+1+block_size] for i in ix]) # stack batch_size=4 arrays cotaining block_size=8 elements from data starting from index+1 in ix

    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    """
    returns the train and val losses over eval_iters iterations
    runs model over a batch of 4 arrays (batch_size) of 8 characters (block_size)
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)   # (n_embd, C=head_size)
        self.query = nn.Linear(n_embd, head_size, bias=False) # (n_embd, C=head_size)
        self.value = nn.Linear(n_embd, head_size, bias=False) # (n_embd, C=head_size)
        # register buffers are not considered as model parameters
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # (T, T)

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)
        v = self.value(x) # (B, T, C)

        # NOTE: C and head_size are equal in our case
        wei = q @ k.transpose(-2, -1) * C**(-0.5) # (B, T, C) @ (B, C, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        out = wei @ v # (B, T, T) @ (B, T, C) ---> (B, T, C)

        return out # (B, T, C)

class BigramLangModel(nn.Module):

    def __init__(self):
        super().__init__()

        # this layer provides the embedding for the token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # this layer provides the embedding for the position of the token
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # keeping the head_size same as n_embd, generally head_size is of lower dimension than n_embd
        self.sa_head = Head(head_size=n_embd)
        # linear layer is to convert from n_embd to vocab_size
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # B, T, C - Batch, Time (one char in a block), Channel (embedding)
        # idx and targets are both (B, T) size inputs

        B, T = idx.shape

        # get the token embeddings from the embedding table
        tok_emb = self.token_embedding_table(idx) # (B, T, C=n_embd)

        # get the position embeddings for 0 to T-1. block_size and T are same. 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C=n_embd)

        x = tok_emb + pos_emb # (B, T, C=n_embd) -- by broadcasting

        # apply one head of self attention
        x = self.sa_head(x) # (B, T, C)
        
        # logits are predictions/scores of next character and here it is simply the embedding of input
        logits = self.lm_head(x) # (B, T, vocab_size)


        if targets is None:
            loss = None

        else:
            # logits and targets need to be reshaped for cross entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # max_new_tokens - number of characters to generate
        # idx - (B, T)
        for _ in range(max_new_tokens):
            # in the first iteration the tokens will be less than or equal to block size
            # but after the first iteration the the tokens will keep on increasing (as we're concatenating in the end)
            # so we need to crop the idx because in the forward method we're generating the positional embeddings
            # if idx is more than block size, then position_embedding_table will run out of scope because it has embeddings upto block_size
            idx_cropped = idx[:, -block_size:] # (B, T)
            # this will return the logits for all the tokens in idx
            logits, loss = self.forward(idx_cropped) # (B, T, C)
            # take out the logits for only the last token because we'll generate the next char only on the basis of last char
            logits = logits[:, -1, :] # (B, C)
            # get the probs from logits
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the prob distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # concatenate for output
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            # next character will be found out on the basis of last character till the loop runs
        return idx
    
model = BigramLangModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):

    # print losses once in a while
    if iter%eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))