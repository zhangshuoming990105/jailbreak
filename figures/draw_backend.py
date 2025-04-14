import matplotlib.pyplot as plt
import numpy as np

# two value, ASR and StrongREJECT score
vllm_xgrammar = [0.992, 0.951]
vllm_outlines = [0.987, 0.884]

sglang_xgrammar = [0.994, 0.926]
sglang_outlines = [0.979, 0.909]

# 修改：创建两个子图
width = 0.15  # 柱状图宽度
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
metrics = ["ASR", "StrongREJECT"]
scores = [
    [vllm_xgrammar[0], vllm_outlines[0], sglang_xgrammar[0], sglang_outlines[0]],
    [vllm_xgrammar[1], vllm_outlines[1], sglang_xgrammar[1], sglang_outlines[1]]
]
labels = ["VLLM XGrammar", "VLLM Outlines", "SGLang XGrammar", "SGLang Outlines"]

for i, ax in enumerate(axs):
    # 绘制每个子图上的四个柱状图（只有一个柱状图组）
    x_pos = np.array([0])
    r1 = ax.bar(x_pos - width, scores[i][0], width, label=labels[0])
    r2 = ax.bar(x_pos,         scores[i][1], width, label=labels[1])
    r3 = ax.bar(x_pos + width,   scores[i][2], width, label=labels[2])
    r4 = ax.bar(x_pos + 2*width, scores[i][3], width, label=labels[3])
    ax.set_title(metrics[i], fontsize=16)  # 设置子图标题为指标
    ax.set_xticks(x_pos)
    ax.set_xticklabels([""])  # 不需要显示 x 刻度标签
    ax.tick_params(axis="both", labelsize=14)
    ax.set_ylim(0, 1.05)
    # 数值注解函数（共用）
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12)
    autolabel(r1)
    autolabel(r2)
    autolabel(r3)
    autolabel(r4)

# 添加共用 y 轴标签
# fig.text(0.04, 0.5, 'Scores', va='center', rotation='vertical', fontsize=16)
# 共用图例，放在图表下方居中
fig.legend(labels, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=12)
plt.suptitle("Enum Attack on Llama-3.1-8B with different backends", fontsize=16)
# 调整布局以显示图例
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("figures/backend_diff.pdf", bbox_inches="tight")