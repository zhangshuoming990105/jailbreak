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
    'openrouter/openai/gpt-5-mini',
    'openrouter/google/gpt-5',
    # 'openrouter/google/gemini-2.0-flash-001',
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
            "end_id": 520  # small for testing
        }
        # Convert model name from "openai/gpt-4o-mini" to "openrouter_openai_gpt-4o-mini"
        filename_model_part = model_name.replace('/', '_')
        filename = f"{dataset_name}_{filename_model_part}.json"
        with open(filename, "w") as fp:
            json.dump(config, fp, indent=2)
