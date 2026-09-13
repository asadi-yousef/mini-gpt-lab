import torch


#loading the whole shakespeare text
with open('data/input.txt','r',encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]   # encoder takes a string and output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])   # decoder takes a list of integers and output a string

# encoding the entire text dataset and store it into a torch.tensor
data = torch.tensor(encode(text),dtype=torch.long)

# train-validation split , 90% train 10% validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8   # how many independent sequences will we process in parallel
batch_size = 4   # maximum context length for predictions

def get_batch(split) :
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0,len(data)-block_size , (batch_size,))     # (batch_size ,)
    x = torch.stack([data[i:i+block_size]] for i in ix)            # (batch_size , block_size)
    y = torch.stack([data[i+1:i+block_size+1]] for i in ix)        # (batch_size , block_size)