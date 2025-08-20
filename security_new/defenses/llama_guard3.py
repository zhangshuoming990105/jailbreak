# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-Guard-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-Guard-3-8B")

def audit(text, tokenizer, model):
    messages = [{"role": "user", "content": text}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=40)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])) # print new generated tokens

# read to_be_audit_grammar.txt
audit_grammar = open("to_be_audit_grammar.txt", "r").read()
audit_prompt = open("to_be_audit_prompt.txt", "r").read()

audit(audit_grammar, tokenizer, model) # safe
audit(audit_prompt, tokenizer, model) # safe

# however, if concatenate them together, it seems to be unsafe
audit(audit_grammar + audit_prompt, tokenizer, model) # unsafe

