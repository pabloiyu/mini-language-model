"""
Created conda environment mistral for this to work.

https://www.youtube.com/watch?v=jqQQWvzo3MM

https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
"""
from ctransformers import AutoModelForCausalLM, AutoTokenizer

llm = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id = "mistral-7b-instruct-v0.1.Q5_K_M.gguf", model_file="mistral-7b-instruct-v0.1.Q5_K_M.gguf", gpu_layers=500)
llm.config.context_length = 4096

# Initialize the conversation history
conversation_history = []

while True:
    user_input = input("\nYou: ")

    # Add the user input to the conversation history
    conversation_history.append(user_input)

    # Generate a response from the model
    input_text = "\n".join(conversation_history)
    response = llm(input_text, max_new_tokens=2048, temperature=0.5, repetition_penalty=1.2) # top_k=55, top_p=0.9,
    print(response)

    # Add the model's response to the conversation history
    conversation_history.append("Model: " + response)
