from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from openai import OpenAI
from loguru import logger

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-Guard-3-8B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-Guard-3-8B")

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
    result_str = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    # check if result_str is safe
    if "unsafe" in result_str.lower():
        return False
    else:
        return True
    
def audit_omni(text) -> bool:
    client = OpenAI()
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )   
    print(response.results[0].flagged)
    if response.results[0].flagged:
        print("--------------------------------")
        print(response.results[0].categories)
        print(response.results[0].category_scores)
        print(response.results[0].category_applied_input_types)
    return not response.results[0].flagged

ds = load_dataset("SafeMTData/SafeMTData",'Attack_600')["Attack_600"]
print(len(ds))

safe_plain = []
safe_actorbreaker = []
unsafe_plain = []
unsafe_actorbreaker = []

for i in range(10):
    item = ds[i]
    plain_query = item["plain_query"]
    actorbreaker_query_list = item["multi_turn_queries"]
    # save the list to a string for audit
    actorbreaker_query_str = "\n".join(actorbreaker_query_list)
    print(f"{i} plain_query: {plain_query}")
    # is_safe = audit(plain_query, tokenizer, model)
    is_safe = audit_omni(plain_query)
    if is_safe:
        print(f"{i} plain_query is safe")
        safe_plain.append(i)
    else:
        print(f"{i} plain_query is unsafe")
        unsafe_plain.append(i)
    print(f"{i} ActorBreaker Query: {actorbreaker_query_str}")
    # is_safe = audit(actorbreaker_query_str, tokenizer, model)
    is_safe = audit_omni(actorbreaker_query_str)
    if is_safe:
        print(f"{i} ActorBreaker Query is safe")
        safe_actorbreaker.append(i)
    else:
        print(f"{i} ActorBreaker Query is unsafe")
        unsafe_actorbreaker.append(i)
    print("--------------------------------")

# print(f"safe_plain: {safe_plain}")
# print(f"unsafe_plain: {unsafe_plain}")
print(f"safe_plain: {len(safe_plain)}")
print(f"unsafe_plain: {len(unsafe_plain)}")
print(f"safe_actorbreaker: {len(safe_actorbreaker)}")
print(f"unsafe_actorbreaker: {len(unsafe_actorbreaker)}")