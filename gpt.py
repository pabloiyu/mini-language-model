with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Length of dataset: {len(text)} characters.")

# There are a total of 65 unique characters in the dataset.
chars = sorted(list(set(text)))
print(len(chars))
print("".join(chars))

# We will tokenize our vocabulary by building a character level language model. We will represent each
# character as an integer. Sub-word tokenizers are also possible (chat-gpt uses tiktoken)
# We first create a mapping from characters to integers using a dictionary
chtoi = {ch:i for i,ch in enumerate(chars)}
itoch = {i:ch for i,ch in enumerate(chars)}

def encode(s):  
    return [chtoi[ch] for ch in s] # Take a string, output list of integers.

def decode(list_int):
    return "".join([itoch[i] for i in list_int]) # Take a list of integers, output string.

# We now encode entire "input.txt" and save it in a torch tensor.
import torch
data = torch.tensor(encode(text))

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# When we train a transformer, we only work with random chunks we take from the dataset. 
#
# In a chunk of 9 characters, there are 8 training examples of increasing context length. Maximum context length
# we train with is given by block_size. This is useful for inference as the transformer is used to working with 
# varying context lengths. For inference, we have to divide inputs larger than block_size into chunks. 
block_size = 8

print("CONTEXT")
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When {context} is the context, the target is {target}.")
    
    

