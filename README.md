# Mini Language Model

## Mamba

Implementing [Mamba SSM](https://arxiv.org/abs/2312.00752) instead of attention heads into a mini language model. For my code to run correctly, it must be ran on a Linux environment. I have tested it and it also functions correctly within a WSL2 environment.

* "llm_mamba_train.py" Trains a mini language model (25M parameters) on the open domain works of Sherlock Holmes and saves resulting LLM model. Training should run on a 8GB VRAM GPU.
* "llm_mamba_use.py" Uses trained LLM model to generate new tokens and saves it to "output.txt".

## Adapter

Similar code structure as Mamba. Implements architecture for incorporating parallel adapter layers into a transformer. Adapters are particularly useful when dealing with large pre-trained language models as they allow us to leverage the knowledge captured in these large models, while only needing to train a relatively small number of parameters.

* "add_adapters.py" Loads state dictionary of trained LLM model, freezes original parameters and adds adapter layers.

## Mistral

Contains a very simple use case of a quantized Mistral-7B model. Also experimenting with vector similarity and simple RAG implementation.
