"""
This script implements an LLM fine-tuning process with adapter modules.  
Key concepts:

* LLM Modification: Introduces adapter layers within the blocks of an existing LLM architecture.
* Selective Freezing: Loads a pre-trained LLM and freezes all weights except for 
the newly added adapter parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from llm_train import Block, LLM, n_embed, n_layers, vocab_size, block_size, encode, n_heads, \
                      get_batch, decode, estimate_loss, max_iters, eval_interval, eval_iters, lr

device = "cuda" if torch.cuda.is_available() else "cpu"

class Adapter(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Adapter, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.adapter_ffwd = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, x):
        return self.adapter_ffwd(x)


class Block_mod(Block):
    # Inherit from old block but add adapter
    def __init__(self, n_embed, n_heads):
        super().__init__(n_embed, n_heads)
        self.adapter = Adapter(n_embed, n_embed//4, n_embed)
        
    def forward(self, x):
        x = super().forward(x)
        x = self.adapter(x)
        return x
            
class LLM_mod(LLM):
    def __init__(self, vocab_size):
        super().__init__(vocab_size)
        self.blocks = nn.Sequential(*[Block_mod(n_embed, n_heads) for _ in range(n_layers)])
        
    def load_pretrained_weights(self, llm):
        self.load_state_dict(llm.state_dict(), strict=False)
        
def main():
    with open('sherlock.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    data = torch.tensor(encode(text))
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # We create the modified model based on the original
    model = LLM_mod(vocab_size).to(device)
    
    # Load the saved model's state dictionary
    saved_model_path = "saved_model.pth"
    model.load_state_dict(torch.load(saved_model_path), strict=False)
    
    # Freeze everything except the adapter parameters
    for name, param in model.named_parameters():
        if "adapter" not in name:  
            param.requires_grad = False

    data = torch.tensor(encode(text))
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Define the optimizer and only include trainable parameters
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Number of trainable parameters: {round(num_params,3)} million")

    for iter in tqdm(range(max_iters)):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, eval_iters, train_data, val_data)
            print(f"iter: {iter}  train_loss: {losses['train']:.4f}  val_loss: {losses['val']:.4f}")
            
        xb, yb = get_batch("train", train_data, val_data)
        logits, loss = model(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    model.eval()
    context = '"You have a case, Holmes?" I remarked.'
    context = torch.tensor(encode(context)).unsqueeze(0).to(device)
    print(context.shape)

    textt = decode(model.generate(context, 2000)[0].tolist())
    print(textt)
    
    
if __name__ == "__main__":
    main()  