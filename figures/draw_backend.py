import matplotlib.pyplot as plt
import numpy as np

# ASR scores only
vllm_xgrammar_asr = 0.992
vllm_outlines_asr = 0.987
sglang_xgrammar_asr = 0.994
sglang_outlines_asr = 0.979

# 创建单个图表，只显示 ASR
width = 0.6  # 增大柱状图宽度，让柱子更紧凑
fig, ax = plt.subplots(figsize=(11, 4))

asr_scores = [vllm_xgrammar_asr, vllm_outlines_asr, sglang_xgrammar_asr, sglang_outlines_asr]
labels = ["VLLM XGrammar", "VLLM Outlines", "SGLang XGrammar", "SGLang Outlines"]

# 绘制四个柱状图，调整间距让柱子更紧凑
x_pos = np.arange(len(labels)) * 0.8  # 减小间距系数
bars = ax.bar(x_pos, asr_scores, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

ax.set_title("ASR - Enum Attack on Llama-3.1-8B with different backends", fontsize=20)  # 增大标题字体
ax.set_xlabel("Backends", fontsize=18)  # 增大轴标签字体
ax.set_ylabel("ASR Score", fontsize=18)  # 增大轴标签字体
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=16)  # 增大刻度标签字体
ax.tick_params(axis="both", labelsize=16)  # 增大刻度字体
ax.set_ylim(0, 1.1)

# 数值注解
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=16)  # 增大数值标注字体

autolabel(bars)

plt.tight_layout()
plt.savefig("figures/backend_diff_asr.pdf", bbox_inches="tight")