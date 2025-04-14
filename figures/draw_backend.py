import matplotlib.pyplot as plt
import numpy as np

# two value, ASR and StrongREJECT score
vllm_xgrammar = [0.992, 0.951]
vllm_outlines = [0.987, 0.884]

sglang_xgrammar = [0.994, 0.926]
sglang_outlines = [0.979, 0.909]

# draw a bar chart with two bars, range from 0 to 1
# in 4 settings together in a chart
x = np.arange(2)  # the label locations
width = 0.15  # the width of the bars
fig, ax = plt.subplots(figsize=(11, 5))  # 修改为扁一点的图形
rects1 = ax.bar(x - width, vllm_xgrammar, width, label="VLLM XGrammar")
rects2 = ax.bar(x, vllm_outlines, width, label="VLLM Outlines")
rects3 = ax.bar(x + width, sglang_xgrammar, width, label="SGLang XGrammar")
rects4 = ax.bar(x + 2 * width, sglang_outlines, width, label="SGLang Outlines")
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Scores", fontsize=16)  # 修改为较大字体
ax.set_title("Enum Attack on Llama-3.1-8B with different backends", fontsize=16)  # 修改为较大字体
ax.set_xticks(x + width / 2)
ax.set_xticklabels(["ASR", "StrongREJECT"])
ax.tick_params(axis="both", labelsize=14)  # 设置坐标轴刻度字体大小
# 调整图例放置在图表下方居中，不与图形重叠
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=12)

ax.set_ylim(0, 1.05)
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 1),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12)  # 设置数值注解的字体大小
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
plt.savefig("figures/backend_diff.pdf")