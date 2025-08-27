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

import json

# dictattack
for dataset_name in dataset_list:
    for model_name in model_list:
        method = 'dictattack'
        config = {
            "model": model_name,
            "dataset_name": dataset_name,
            "method": method,
        }
        filename = f"{method}_{dataset_name}_{model_name}.json"
        with open(filename, "w") as fp:
            json.dump(config, fp)
