from util import MODEL_TYPE_MAP, get_client, get_logger
import json
import os
from datasets import Dataset, load_dataset

model_name = "meta-llama/Llama-3.1-8B-Instruct"
client = get_client(MODEL_TYPE_MAP[model_name])
logger = get_logger("INFO", f"official_logs/llama3.1_jbshield.log")

ijp_dataset_path = "/home/zhangshuoming/workspace/jailbreak/3rdparty/JBShield/data/jailbreak/ijp/llama-3.1_test.json"

base64_dataset_path = "/home/zhangshuoming/workspace/jailbreak/3rdparty/JBShield/data/jailbreak/base64/llama-3.1_test.json"

drattack_dataset_path = "/home/zhangshuoming/workspace/jailbreak/3rdparty/JBShield/data/jailbreak/drattack/llama-3.1_test.json"

puzzler_dataset_path = "/home/zhangshuoming/workspace/jailbreak/3rdparty/JBShield/data/jailbreak/puzzler/llama-3.1_test.json"

zulu_dataset_path = "/home/zhangshuoming/workspace/jailbreak/3rdparty/JBShield/data/jailbreak/zulu/llama-3.1_test.json"

# get from JBShield-D_llama-3.1.log
ijp_escape_id = [174, 312, 319, 327, 477, 726, 795] # total 820

base64_escape_id = [72, 92, 110, 123] # total 490

drattack_escape_id = [] # total 20

puzzler_escape_id = [] # total 820

zulu_escape_id = [] # total 820

# 读取 JSON 文件
with open(ijp_dataset_path, "r") as f:
    ijp_data = json.load(f)
with open(base64_dataset_path, "r") as f:
    base64_data = json.load(f)
with open(drattack_dataset_path, "r") as f:
    drattack_data = json.load(f)
with open(puzzler_dataset_path, "r") as f:
    puzzler_data = json.load(f)
with open(zulu_dataset_path, "r") as f:
    zulu_data = json.load(f)


jb_ijp_prompts = [item["jailbreak"] for item in ijp_data]
jb_ijp_questions = [item["jailbreak"] for item in ijp_data]
jb_ijp_prompts_test = jb_ijp_prompts
jb_ijp_questions_test = jb_ijp_questions

ds = Dataset.from_dict({
    "prompt": jb_ijp_prompts_test,
    "goal": jb_ijp_questions_test,
})
# ds.push_to_hub("mistral0105/jbshield_ijp")
    
# for i in range(len(jb_ijp_prompts_test)):
#     goal = jb_ijp_questions_test[i]
#     logger.info(f"id: {ijp_escape_id[i]}")
#     logger.info(f"Goal: {goal}")
#     prompt = jb_ijp_prompts_test[i]
#     logger.info(f"Prompt: {prompt}")
#     logger.info("==="*20)
#     # call llm
#     response = client.chat.completions.create(
#         messages=[
#             {"role": "user", "content": prompt},
#         ],
#         model=model_name,
#         temperature=0,
#         max_tokens=4096,
#     )
#     logger.info(f"Response: {response.choices[0].message.content}")

jb_base64_prompts = [item["jailbreak"] for item in base64_data]
jb_base64_questions = [item["goal"] for item in base64_data]
jb_base64_prompts_test = jb_base64_prompts
jb_base64_questions_test = jb_base64_questions

# save to dataset and push to hub
ds = Dataset.from_dict({
    "prompt": jb_base64_prompts_test,
    "goal": jb_base64_questions_test,
})
# ds.push_to_hub("mistral0105/jbshield_base64")

# for i in range(len(jb_base64_prompts_test)):
#     goal = jb_base64_questions_test[i]
#     logger.info("==="*20)
#     logger.info(f"id: {base64_escape_id[i]}")
#     logger.info(f"Goal: {goal}")
#     prompt = jb_base64_prompts_test[i]
#     logger.info(f"Prompt: {prompt}")
#     # call llm
#     response = client.chat.completions.create(
#         messages=[
#             {"role": "user", "content": prompt},
#         ],
#         model=model_name,
#         temperature=0,
#         max_tokens=4096,
#     )
#     logger.info(f"Response: {response.choices[0].message.content}")

jb_drattack_prompts = [item["jailbreak"] for item in drattack_data]
jb_drattack_questions = [item["goal"] for item in drattack_data]

ds = Dataset.from_dict({
    "prompt": jb_drattack_prompts,
    "goal": jb_drattack_questions,
})
ds.push_to_hub("mistral0105/jbshield_drattack")

jb_zulu_prompts = [item["jailbreak"] for item in zulu_data]
jb_zulu_questions = [item["goal"] for item in zulu_data]

ds = Dataset.from_dict({
    "prompt": jb_zulu_prompts,
    "goal": jb_zulu_questions,
})
ds.push_to_hub("mistral0105/jbshield_zulu")

jb_puzzler_prompts = [item["jailbreak"] for item in puzzler_data]
jb_puzzler_questions = [item["goal"] for item in puzzler_data]

ds = Dataset.from_dict({
    "prompt": jb_puzzler_prompts,
    "goal": jb_puzzler_questions,
})
ds.push_to_hub("mistral0105/jbshield_puzzler")
