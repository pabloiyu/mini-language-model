import torch
import torch.nn as nn
from torch.nn import functional as F
from llm_train import LLM, vocab_size, encode, decode
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Define your model architecture and initialize it
model = LLM(vocab_size).to(device)

# Load the saved model's state dictionary
saved_model_path = "saved_model.pth"
model.load_state_dict(torch.load(saved_model_path, map_location=device))
model.eval()  # Set the model to evaluation mode

# We now provide context and let the model generate tokens
context = '"You have a case, Holmes?" I remarked.'
context = torch.tensor(encode(context)).unsqueeze(0).to(device)
print(context.shape)

t1 = time.time()
text = decode(model.generate(context, 200)[0].tolist())
t2 = time.time()

print(text)
print(f"Time taken: {t2-t1:.2f} seconds")

with open("output.txt", "w", encoding="utf-8") as file:
    file.write(text)