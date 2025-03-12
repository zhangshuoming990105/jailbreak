# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B")
model = AutoModelForCausalLM.from_pretrained("Qwen/QwQ-32B", resume_download=True)