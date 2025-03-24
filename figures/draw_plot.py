import  matplotlib.pyplot as plt
import numpy as np

# ASR

## baselines
asr_gpt4o_advbench_base = 0.011538461538461539
asr_gpt4omini_advbench_base = 0.021153846153846155
asr_phi35moe_advbench_base = 0.11538461538461539
asr_gemini2flash_advbench_base = 0.175

asr_gpt4o_harmbench_base = 0.26
asr_gpt4omini_harmbench_base = 0.44
asr_phi35moe_harmbench_base = 0.31
asr_gemini2flash_harmbench_base = 0.46

asr_gpt4o_jailbreakbench_base = 0.1
asr_gpt4omini_jailbreakbench_base = 0.14
asr_phi35moe_jailbreakbench_base = 0.21
asr_gemini2flash_jailbreakbench_base = 0.22

asr_gpt4o_sorrybench_base = 0.3386363636363636
asr_gpt4omini_sorrybench_base = 0.425
asr_phi35moe_sorrybench_base = 0.37727272727272726
asr_gemini2flash_sorrybench_base = 0.29318181818181815

asr_gpt4o_strongreject_base = 0.051118210862619806
asr_gpt4omini_strongreject_base = 0.08945686900958466
asr_phi35moe_strongreject_base = 0.09904153354632587
asr_gemini2flash_strongreject_base = 0.06070287539936102

## ours
asr_gpt4o_advbench_cda = 1.0
asr_gpt4omini_advbench_cda = 0.9903846153846154
asr_phi35moe_advbench_cda = 0.9923076923076923
asr_gemini2flash_advbench_cda = 0.9576923076923077

asr_gpt4o_harmbench_cda = 1.0
asr_gpt4omini_harmbench_cda = 1.0
asr_phi35moe_harmbench_cda = 0.98
asr_gemini2flash_harmbench_cda = 0.92

asr_gpt4o_jailbreakbench_cda = 1.0
asr_gpt4omini_jailbreakbench_cda = 0.97
asr_phi35moe_jailbreakbench_cda = 0.95
asr_gemini2flash_jailbreakbench_cda = 0.92

asr_gpt4o_sorrybench_cda = 0.9636363636363636
asr_gpt4omini_sorrybench_cda = 0.9636363636363636
asr_phi35moe_sorrybench_cda = 0.9840909090909091
asr_gemini2flash_sorrybench_cda = 0.8954545454545455

asr_gpt4o_strongreject_cda = 0.9936102236421726
asr_gpt4omini_strongreject_cda = 0.9808306709265175
asr_phi35moe_strongreject_cda = 0.9648562300319489
asr_gemini2flash_strongreject_cda = 0.8115015974440895

# StrongREJECT score

## baselines
score_gpt4o_advbench_base = 0.010817307692307692
score_gpt4omini_advbench_base = 0.017548076923076923
score_phi35moe_advbench_base = 0.10576923076923077
score_gemini2flash_advbench_base = 0.13028846153846155

score_gpt4o_harmbench_base = 0.2225
score_gpt4omini_harmbench_base = 0.39125
score_phi35moe_harmbench_base = 0.27375
score_gemini2flash_harmbench_base = 0.4125

score_gpt4o_jailbreakbench_base = 0.09125
score_gpt4omini_jailbreakbench_base = 0.13125
score_phi35moe_jailbreakbench_base = 0.19125
score_gemini2flash_jailbreakbench_base = 0.1975

score_gpt4o_sorrybench_base = 0.3039772727272727
score_gpt4omini_sorrybench_base = 0.39801136363636364
score_phi35moe_sorrybench_base = 0.3346590909090909
score_gemini2flash_sorrybench_base = 0.26732954545454546

score_gpt4o_strongreject_base = 0.04752396166134185
score_gpt4omini_strongreject_base = 0.08027156549520767
score_phi35moe_strongreject_base = 0.08746006389776358
score_gemini2flash_strongreject_base = 0.04233226837060703

## ours
score_gpt4o_advbench_cda = 0.9557692307692308
score_gpt4omini_advbench_cda = 0.9149038461538461
score_phi35moe_advbench_cda = 0.7694711538461538
score_gemini2flash_advbench_cda = 0.9317307692307693

score_gpt4o_harmbench_cda = 0.9075
score_gpt4omini_harmbench_cda = 0.865
score_phi35moe_harmbench_cda = 0.74625
score_gemini2flash_harmbench_cda = 0.855

score_gpt4o_jailbreakbench_cda = 0.905
score_gpt4omini_jailbreakbench_cda = 0.85875
score_phi35moe_jailbreakbench_cda = 0.7225
score_gemini2flash_jailbreakbench_cda = 0.85

score_gpt4o_sorrybench_cda = 0.7960227272727273
score_gpt4omini_sorrybench_cda = 0.7585227272727273
score_phi35moe_sorrybench_cda = 0.7099431818181818
score_gemini2flash_sorrybench_cda = 0.7539772727272728

score_gpt4o_strongreject_cda = 0.8777955271565495
score_gpt4omini_strongreject_cda = 0.8522364217252396
score_phi35moe_strongreject_cda = 0.6777156549520766
score_gemini2flash_strongreject_cda = 0.7480031948881789


# 新增：绘图代码，比较5个 benchmark 上4种模型的 ASR（baseline vs CDA）
import matplotlib.pyplot as plt
import numpy as np

# 修改后：在同一子图上展示 ASR 柱状图和 StrongREJECT score 折线图
models = ["gpt-4o", "4o-mini", "phi-3.5", "gemini-2.0"]

benchmarks = {
    "AdvBench": {
        "base": [asr_gpt4o_advbench_base, asr_gpt4omini_advbench_base, asr_phi35moe_advbench_base, asr_gemini2flash_advbench_base],
        "cda":  [asr_gpt4o_advbench_cda, asr_gpt4omini_advbench_cda, asr_phi35moe_advbench_cda, asr_gemini2flash_advbench_cda]
    },
    "HarmBench": {
        "base": [asr_gpt4o_harmbench_base, asr_gpt4omini_harmbench_base, asr_phi35moe_harmbench_base, asr_gemini2flash_harmbench_base],
        "cda":  [asr_gpt4o_harmbench_cda, asr_gpt4omini_harmbench_cda, asr_phi35moe_harmbench_cda, asr_gemini2flash_harmbench_cda]
    },
    "JailbreakBench": {
        "base": [asr_gpt4o_jailbreakbench_base, asr_gpt4omini_jailbreakbench_base, asr_phi35moe_jailbreakbench_base, asr_gemini2flash_jailbreakbench_base],
        "cda":  [asr_gpt4o_jailbreakbench_cda, asr_gpt4omini_jailbreakbench_cda, asr_phi35moe_jailbreakbench_cda, asr_gemini2flash_jailbreakbench_cda]
    },
    "SorryBench": {
        "base": [asr_gpt4o_sorrybench_base, asr_gpt4omini_sorrybench_base, asr_phi35moe_sorrybench_base, asr_gemini2flash_sorrybench_base],
        "cda":  [asr_gpt4o_sorrybench_cda, asr_gpt4omini_sorrybench_cda, asr_phi35moe_sorrybench_cda, asr_gemini2flash_sorrybench_cda]
    },
    "StrongREJECT": {
        "base": [asr_gpt4o_strongreject_base, asr_gpt4omini_strongreject_base, asr_phi35moe_strongreject_base, asr_gemini2flash_strongreject_base],
        "cda":  [asr_gpt4o_strongreject_cda, asr_gpt4omini_strongreject_cda, asr_phi35moe_strongreject_cda, asr_gemini2flash_strongreject_cda]
    }
}

# 新增：StrongREJECT score 数据字典
strongreject = {
    "AdvBench": {
        "base": [score_gpt4o_advbench_base, score_gpt4omini_advbench_base, score_phi35moe_advbench_base, score_gemini2flash_advbench_base],
        "cda":  [score_gpt4o_advbench_cda, score_gpt4omini_advbench_cda, score_phi35moe_advbench_cda, score_gemini2flash_advbench_cda]
    },
    "HarmBench": {
        "base": [score_gpt4o_harmbench_base, score_gpt4omini_harmbench_base, score_phi35moe_harmbench_base, score_gemini2flash_harmbench_base],
        "cda":  [score_gpt4o_harmbench_cda, score_gpt4omini_harmbench_cda, score_phi35moe_harmbench_cda, score_gemini2flash_harmbench_cda]
    },
    "JailbreakBench": {
        "base": [score_gpt4o_jailbreakbench_base, score_gpt4omini_jailbreakbench_base, score_phi35moe_jailbreakbench_base, score_gemini2flash_jailbreakbench_base],
        "cda":  [score_gpt4o_jailbreakbench_cda, score_gpt4omini_jailbreakbench_cda, score_phi35moe_jailbreakbench_cda, score_gemini2flash_jailbreakbench_cda]
    },
    "SorryBench": {
        "base": [score_gpt4o_sorrybench_base, score_gpt4omini_sorrybench_base, score_phi35moe_sorrybench_base, score_gemini2flash_sorrybench_base],
        "cda":  [score_gpt4o_sorrybench_cda, score_gpt4omini_sorrybench_cda, score_phi35moe_sorrybench_cda, score_gemini2flash_sorrybench_cda]
    },
    "StrongREJECT": {
        "base": [score_gpt4o_strongreject_base, score_gpt4omini_strongreject_base, score_phi35moe_strongreject_base, score_gemini2flash_strongreject_base],
        "cda":  [score_gpt4o_strongreject_cda, score_gpt4omini_strongreject_cda, score_phi35moe_strongreject_cda, score_gemini2flash_strongreject_cda]
    }
}

# 定义2行3列子图布局，共6个子图
width = 0.35
x = np.arange(len(models))
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

bench_list = list(benchmarks.items())
for idx, (bench, data) in enumerate(bench_list):
    ax = axs[idx]
    # 绘制 ASR 柱状图
    bars_base = ax.bar(x - width/2, data["base"], width, color="tab:blue")   # ASR Baseline
    bars_cda = ax.bar(x + width/2, data["cda"], width, color="tab:orange")    # ASR CDA
    ax.set_title(bench)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("ASR")
    ax.set_ylim(0, 1)  # 统一 ASR y 轴范围
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # 修改数值标注的位置，增加一个垂直偏移量避免重叠
    for bar in bars_base:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=10, color='tab:blue')
    for bar in bars_cda:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=10, color='tab:orange')
    
    # 右侧 y 轴绘制 StrongREJECT Score 折线图，使用不同颜色
    ax2 = ax.twinx()
    s_base = strongreject[bench]["base"]
    s_cda = strongreject[bench]["cda"]
    line1 = ax2.plot(x, s_base, marker='s', linestyle='--', color='tab:green')[0]   # StrongREJECT Baseline
    line2 = ax2.plot(x, s_cda, marker='^', linestyle='--', color='tab:red')[0]       # StrongREJECT CDA
    ax2.set_ylabel("StrongREJECT Score")
    ax2.set_ylim(0, 1)  # 统一 StrongREJECT y 轴范围
    
    # 标注 StrongREJECT 折线图的数据点
    for xi, yi in zip(x, s_base):
        ax2.text(xi, yi + 0.02, f'{yi:.3f}', ha='center', va='bottom', fontsize=10, color='tab:green')
    for xi, yi in zip(x, s_cda):
        ax2.text(xi, yi + 0.02, f'{yi:.3f}', ha='center', va='bottom', fontsize=10, color='tab:red')
    
# 将第6个子图作为图例区（原空白）
axs[-1].clear()
axs[-1].axis('off')
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='tab:blue', label='ASR Baseline'),
    Patch(facecolor='tab:orange', label='ASR CDA'),
    Line2D([0], [0], marker='s', color='tab:green', linestyle='--', label='StrongREJECT Baseline'),
    Line2D([0], [0], marker='^', color='tab:red', linestyle='--', label='StrongREJECT CDA'),
]
# 调大图例的字体大小以提高可读性
axs[-1].legend(handles=legend_elements, loc='center', fontsize=18)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig("asr_comparison.pdf")
plt.show()

