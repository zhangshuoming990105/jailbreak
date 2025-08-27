dataset_list = [
    'advbench',
    'harmbench',
    'SorryBench',
    'JailbreakBench',
    'strongreject',
]

model_list = [
    'gpt-4o',
    'gpt-4o-mini',
    'gemini-2.0-flash'
]

def client_type(model: str):
    if model.startswith('gpt'):
        return 'openai'
    elif model.startswith('gemini'):
        return 'gemini'

import os, json

for dataset_name in dataset_list:
    for model_name in model_list:
        model_client = client_type(model_name)

        config = {
            "assemble_client": "openai",
            "assemble_model": "gpt-4o-mini",
            "assemble_num": 1,

            "attack_client": model_client,
            "attack_model": model_name,
            "dataset_name": dataset_name,
        }
        filename = f"{dataset_name}_{model_name}.json"
        with open(filename, "w") as fp:
            json.dump(config, fp)


        
