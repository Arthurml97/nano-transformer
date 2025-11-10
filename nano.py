import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

# Using this to study transformers from scratch.
# Based on the nanoGPT implementation by Andrej Karpathy.
# I not using that comments in production code normally.
# if you dont have a gpu, you cant run this code efficiently, and it will be slow. So make sure you have a cuda-capable gpu (nvidia).

batch_size = 64 # how many indepedent sequences will we process in parallel
block_size = 256 # what is the maximum context length for predictions
max_iters = 5000 # number of training iterations
eval_interval = 500 # how often to evaluate the loss
learning_rate = 1e-4 # learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # number of iterations to estimate loss
n_embd = 384 # the dimensionality of the character embedding vectors
n_head = 6 # number of attention heads
n_layer = 6 # number of transformer blocks
dropout = 0.3 # dropout rate
# ------------------------------

torch.manual_seed(1337) # set the random seed for reproducibility

with open('input.txt', 'r', encoding='utf-8') as f: # read the input text file
    text = f.read() # store the text in a string variable

# tokenizer setup
tokenizer = Tokenizer.from_file("bpe_tokenizer.json") # load the tokenizer from file
vocab_size = tokenizer.get_vocab_size() # get the vocabulary size
print("Vocab size:", vocab_size) # print the vocabulary size
encode = lambda s: tokenizer.encode(s).ids # function to encode a string to a list of token ids
decode = lambda l: tokenizer.decode(l) # function to decode a list of token ids to a string


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)  # encode the entire text dataset and store it in a tensor
# split the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest validation
train_data = data[:n] # training data
val_data = data[n:] # validation data

# data loading
def get_batch(split): # generate a batch of data
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data # select the appropriate dataset
    ix = torch.randint(len(data) - block_size, (batch_size,)) # random starting indices for the batch
    x = torch.stack([data[i:i+block_size] for i in ix]) # input sequences
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # target sequences
    x, y = x.to(device), y.to(device) # move to device
    return x, y # return the input and target batches

@torch.no_grad() # disable gradient tracking for evaluation
def estimate_loss(): # estimate the loss on train and val sets
    out = {} # dictionary to hold the losses
    model.eval() # set the model to evaluation mode
    for split in ['train', 'val']: # evaluate on both train and validation sets
        losses = torch.zeros(eval_iters) # tensor to hold the losses
        for k in range(eval_iters): #  number of evaluation iterations
            X, Y = get_batch(split) # get a batch of data
            logits, loss = model(X, Y) # forward pass
            losses[k] = loss.item() # store the loss
        out[split] = losses.mean() # compute the mean loss
    model.train() # set the model back to training mode
    return out # return the losses

class Head(nn.Module): # single attention head

    def __init__(self, head_size): # initialize the head
        super().__init__() # call the parent class constructor
        self.key = nn.Linear(n_embd, head_size, bias=False) # key projection
        self.query = nn.Linear(n_embd, head_size, bias=False) # query projection
        self.value = nn.Linear(n_embd, head_size, bias=False) # value projection
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lower triangular matrix for masking

        self.dropout = nn.Dropout(dropout) # dropout layer

    def forward(self, x): # forward pass
        B, T, C = x.shape # batch size, time steps, number of channels
        k = self.key(x) # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei) # apply dropout to the attention weights
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out

class MultiHeadAttention(nn.Module): # multi-head attention

        def __init__(self, num_heads, head_size): # initialize the multi-head attention
            super().__init__() # call the parent class constructor
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # list of attention heads
            self.proj = nn.Linear(num_heads * head_size,n_embd) # output projection
            self.dropout = nn.Dropout( dropout) # dropout layer


        def forward(self, x): # forward pass
            out = torch.cat([h.forward(x) for h in self.heads], dim=-1) # concatenate the outputs of the heads
            out = self.proj(out) # project the concatenated output
            return out

class FeedForward(nn.Module): # feedforward neural network

    def __init__(self, n_embd): # initialize the feedforward network
        super().__init__() # call the parent class constructor
        self.net = nn.Sequential( # sequential model
            nn.Linear(n_embd, 4 * n_embd), # first linear layer
            nn.ReLU(), # ReLU activation
            nn.Linear(4 * n_embd, n_embd), # second linear layer
            nn.Dropout(dropout), # dropout layer
        )

    def forward(self, x): # forward pass
        return self.net(x) # pass through the network

class Block(nn.Module): # transformer block

    def __init__(self, n_embd, n_head): # initialize the block
        super().__init__() # call the parent class constructor
        head_size = n_embd // n_head # size of each head
        self.sa = MultiHeadAttention(n_head, head_size) # multi-head attention layer
        self.ffwd = FeedForward(n_embd) # feedforward layer
        self.ln1 = nn.LayerNorm(n_embd) # layer normalization 1
        self.ln2 = nn.LayerNorm(n_embd) # layer normalization 2

    def forward(self, x): # forward pass
        x = x + self.sa(self.ln1(x)) # multi-head attention
        x = x + self.ffwd(self.ln2(x)) # feedforward
        return x

# simple bigram model
class NanoTransformer(nn.Module): # define the bigram language model

    def __init__(self): # initialize the model
        super().__init__() # call the parent class constructor
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # embedding layer
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # positional embedding layer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # transformer blocks (4 heads per block)
        self.ln_f = nn.LayerNorm(n_embd) # final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size) # linear layer to produce logits

    def forward(self, idx, targets=None): # forward pass
        B, T = idx.shape # batch size and time steps

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)py
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,Vocab_size)

        if targets is None: # if no targets provided, return logits only
            loss = None # no loss to compute
        else: # compute the loss
            B, T, C = logits.shape # batch size, time steps, number of classes
            logits = logits.view(B*T, C) # reshape logits to (B*T, C)
            targets = targets.view(B*T) # reshape targets to (B*T)
            loss = F.cross_entropy(logits, targets) # compute cross-entropy loss

        return logits, loss # return logits and loss

    def generate(self, idx, max_new_tokens): # generate new tokens
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens): # generate max_new_tokens tokens
            idx_cond = idx[:, -block_size:] # crop idx to the last block_size tokens
            # get the predictions
            logits, loss = self(idx_cond) # (B,T,C)
            # focus only on the last time step
            logits = logits[:, -1, :] # (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            next_idx = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, next_idx), dim=1) # (B,T+1)
        return idx

model = NanoTransformer () # instantiate the model
m = model.to(device) # move the model to the device
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters') # print the number of parameters in millions

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # AdamW optimizer

for iter in range(max_iters): # training loop

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0: # time to evaluate
        losses = estimate_loss() # estimate the losses
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}") # print the losses

    # sample a batch of data
    xb, yb = get_batch('train') # get a batch of training data

    # evaluate the loss
    logits, loss = model(xb, yb) # forward pass
    optimizer.zero_grad(set_to_none=True) # zero the gradients
    loss.backward() # backward pass
    optimizer.step() # update the parameters

# generate some text after training
start_token_str = "[CLS]" # starting token string
start_token_id = tokenizer.token_to_id(start_token_str) # get the token id for the starting token
if start_token_id is None:
    print(f"Token '{start_token_str}' not found in the tokenizer vocabulary.")
    start_token_id = 0  # default to 0 if token not found
context = torch.tensor([[start_token_id]], dtype=torch.long, device=device) # starting context
print(decode(model.generate(context, max_new_tokens=500)[0].tolist())) # generate and decode 500 new tokens
