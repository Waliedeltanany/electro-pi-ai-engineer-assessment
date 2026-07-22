import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model loaded!")

prompt = "Explain what Retrieval-Augmented Generation (RAG) is in simple terms."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

torch.cuda.reset_peak_memory_stats()

start = time.time()

output = model.generate(
    **inputs,
    max_new_tokens=200
)

end = time.time()

generated = output[0][inputs["input_ids"].shape[1]:]

text = tokenizer.decode(generated, skip_special_tokens=True)

tokens = len(generated)

print("="*60)
print(text)
print("="*60)

print(f"Generation Time : {end-start:.2f} sec")
print(f"Generated Tokens: {tokens}")
print(f"Tokens/sec      : {tokens/(end-start):.2f}")
print(f"Peak VRAM       : {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")