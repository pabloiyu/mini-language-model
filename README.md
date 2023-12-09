# Mini Language Model

## Mamba

Implementing [Mamba SSM](https://arxiv.org/abs/2312.00752) instead of attention heads into a mini language model. For my code to run correctly, it must be ran on a Linux environment. I have tested it and it also functions correctly within a WSL2 environment.

* "llm_mamba_train.py" Trains a mini language model on the open domain works of Sherlock Holmes and saves resulting LLM model.
* "llm_mamba_use.py" Uses trained LLM model to generate new tokens and saves it to "output.txt".

## Mistral

Contains a very simple use case of a quantized Mistral-7B model.
