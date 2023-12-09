import torch
import torch.nn as nn
from torch.nn import functional as F
from llm_mamba_train import LLM, vocab_size, encode, decode

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define your model architecture and initialize it
model = LLM(vocab_size).to(device)

# Load the saved model's state dictionary
saved_model_path = "saved_model.pth"
model.load_state_dict(torch.load(saved_model_path, map_location=device))
model.eval()  # Set the model to evaluation mode

# We now provide context and let the model generate tokens
context = "".join(5*[" "]) + "You have a case, Holmes? I remarked."
context = torch.tensor(encode(context)).unsqueeze(0).to(device)
print(context.shape)

text = decode(model.generate(context, 2000)[0].tolist())
print(text)

with open("output.txt", "w", encoding="utf-8") as file:
    file.write(text)