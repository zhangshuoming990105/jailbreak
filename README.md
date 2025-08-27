# Artifacts of USENIX Sec26 Cycle1 Submission 705

## Critical File structures

- official_logs, saves recording logs for previous run
- cda_attack.py, implements EnumAttack schemas and implementaitons, accelerated through python async API
- dict_attack.py, implements DictAttack dictionary creation, DictAttack key construction and recovered auditing grammar generation, accelerated through python async API
- eval_qa.py, the file used to evaluate the csv results generated from previous runs on the model attack side
- guard.py, the file used to evaluate different mitigation strategies and models for this attack
- analyze_logit_beam.py, case study(Figure 7 and Figure 8) used, which generates logit results of Phi-3.5-MoE model.
- draw_logit_shift.py, case study used, which generates the desired logit results for analyze_logit_beam.py usage.
- main_async.py, major entrypoint for most evaluations.


## Dependencies

## Evaluation commands

## Hardware requirements

