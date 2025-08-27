dataset_list = [
    'advbench',
    # 'harmbench',
    # 'SorryBench',
    # 'JailbreakBench',
    # 'strongreject',
]

eval_methods = ["llamaguard", "moderation", "llm"]
audit_components = ["grammar", "full"]

import json

# enumattack
for dataset_name in dataset_list:
    for eval_method in eval_methods:
        attack_method = "enumattack"
        # audit_component = "grammar"
        config = {
            "dataset_name": dataset_name,
            "eval_method": eval_method,
            "attack_method": attack_method,
            "audit_component": "grammar",
        }
        filename = f"{attack_method}_{dataset_name}_{eval_method}.json"
        with open(filename, "w") as fp:
            json.dump(config, fp)

# dictattack
for dataset_name in dataset_list:
    for eval_method in eval_methods:
        for audit_component in audit_components:
            for assemble_num in [0, 1, 10]:
                attack_method = "dictattack"
                config = {
                    "dataset_name": dataset_name,
                    "eval_method": eval_method,
                    "attack_method": attack_method,
                    "audit_component": audit_component,
                    "dict_assemble": {
                        "model": "gpt-4o",
                        "num": assemble_num
                    }
                }
                filename = f"{attack_method}_{dataset_name}_{audit_component}_k{assemble_num}_{eval_method}.json"
                with open(filename, "w") as fp:
                    json.dump(config, fp)


