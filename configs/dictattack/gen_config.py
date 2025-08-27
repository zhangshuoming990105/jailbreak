import json

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

for dataset_name in dataset_list:
    for model_name in model_list:
        config = {
            "dict_assemble": {
                "model": "gpt-4o-mini",
                "num": 1
            },
            "model": model_name,
            "dataset_name": dataset_name,
        }
        filename = f"{dataset_name}_{model_name}.json"
        with open(filename, "w") as fp:
            json.dump(config, fp)
