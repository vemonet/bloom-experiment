import transformers
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast
import torch


# Download model
model = BloomForCausalLM.from_pretrained("bigscience/bloom")

# Get tokenizer to turn prompts into an embedding Bloom can understand
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b3")


prompt = "It was a dark and stormy night"
result_length = 50
inputs = tokenizer(prompt, return_tensors="pt")


# Greedy Search
print(tokenizer.decode(model.generate(inputs["input_ids"], max_length=result_length)[0]))

# Beam Search
print(tokenizer.decode(model.generate(
    inputs["input_ids"],
    max_length=result_length,
    num_beams=2,
    no_repeat_ngram_size=2,
    early_stopping=True
)[0]))

# Sampling Top-k + Top-p
print(tokenizer.decode(model.generate(
    inputs["input_ids"],
    max_length=result_length,
    do_sample=True,
    top_k=50,
    top_p=0.9
)[0]))