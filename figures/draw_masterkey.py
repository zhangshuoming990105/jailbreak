import matplotlib.pyplot as plt
import numpy as np

# 数据
queries = [1, 5, 10, 20, 50, 100, 200]
ASR_MasterKey = [0.0211, 0.0267, 0.0367, 0.045, 0.054, 0.062, 0.069]
StrongREJECT_MasterKey = [0.0176, 0.019, 0.0252, 0.0304, 0.037, 0.042, 0.048]
ASR_Enum_Attack_MasterKey = [0.99, 1, 1, 1, 1, 1, 1]
StrongREJECT_Enum_Attack_MasterKey = [0.915, 0.946, 0.953, 0.958, 0.967, 0.976, 0.985]

# 创建图表和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 绘制左侧子图：MasterKey数据
line1, = ax1.plot(queries, ASR_MasterKey, 'o-', color='#1f77b4')
line2, = ax1.plot(queries, StrongREJECT_MasterKey, 'o-', color='#ff7f0e')
ax1.set_title('MasterKey Attack\'s ASR and StrongREJECT vs Queries')
ax1.set_xlabel('Queries')
ax1.set_ylabel('Scores')
ax1.grid(True)

# 绘制右侧子图：Enum Attack + MasterKey数据
line3, = ax2.plot(queries, ASR_Enum_Attack_MasterKey, 'o-', color='#1f77b4')
line4, = ax2.plot(queries, StrongREJECT_Enum_Attack_MasterKey, 'o-', color='#ff7f0e')
ax2.set_title('Enum Attack\'s ASR and StrongREJECT vs Queries')
ax2.set_xlabel('Queries')
ax2.set_ylabel('Scores')
ax2.grid(True)

# 创建公共图例
fig.legend(
    [line1, line2], 
    ['ASR', 'StrongREJECT'],
    loc='upper center', 
    bbox_to_anchor=(0.5, 0.1),
    ncol=2
)

plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # 为底部的图例留出空间
plt.savefig('masterkey@k.pdf')
plt.show()
