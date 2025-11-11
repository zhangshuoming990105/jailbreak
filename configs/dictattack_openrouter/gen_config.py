import json

dataset_list = [
    'advbench',
    # 'harmbench',
    # 'SorryBench',
    # 'JailbreakBench',
    # 'strongreject',
]

model_list = [
    'openrouter/openai/gpt-4o',
    'openrouter/openai/gpt-4o-mini',
    # newest flagship models
    'openrouter/openai/gpt-5',
    'openrouter/google/gemini-2.5-pro',
    # structuredoutput supported open-weight models
    'openrouter/deepseek/deepseek-r1-0528',
    'openrouter/meta-llama/llama-4-scout',
    'openrouter/openai/gpt-oss-120b',
    # 'openrouter/qwen/qwen3-30b-a3b-instruct-2507',
    # 'openrouter/qwen/qwen3-235b-a22b-thinking-2507',
    # 'openrouter/qwen/qwen3-32b',
    # 'openrouter/z-ai/glm-4.6',
    # 'openrouter/z-ai/glm-4.5',
    # 'openrouter/moonshotai/kimi-k2-0905',
    # 'openrouter/qwen/qwen3-next-80b-a3b-thinking',
    # 'openrouter/qwen/qwen3-next-80b-a3b-instruct',
    # 'openrouter/moonshotai/kimi-k2-thinking',
    # 'openrouter/minimax/minimax-m2',
    # 'openrouter/google/gemini-2.5-flash',
    # 'openrouter/deepseek/deepseek-v3.1-terminus',
    # 'openrouter/moonshotai/kimi-k2-thinking',
    
]

for dataset_name in dataset_list:
    for model_name in model_list:
        config = {
            "dict_assemble": {
                "model": "openrouter/openai/gpt-4o-mini",
                "num": 1
            },
            "model": f"{model_name}",
            "dataset_name": dataset_name,
            "begin_id": 0,
            "end_id": 3,  # small for testing,
            "success_only": True
        }
        # Convert model name from "openai/gpt-4o-mini" to "openrouter_openai_gpt-4o-mini"
        filename_model_part = model_name.replace('/', '_')
        filename = f"{dataset_name}_{filename_model_part}.json"
        with open(filename, "w") as fp:
            json.dump(config, fp, indent=2)
