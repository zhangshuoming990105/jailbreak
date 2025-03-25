import os
import re
import glob

# 日志文件的目录
log_dir = './official_logs/baseline/sorrybench'
# 日志文件的模式
log_pattern = 'gpt-4o-mini_SorryBench03_baseline_*.log'

# 用于存储每个类别的分数
scores_by_style = {}
count_by_style = {}

# 查找所有匹配的日志文件
log_files = glob.glob(os.path.join(log_dir, log_pattern))

for log_file in log_files:
    # 从文件名提取style名称
    filename = os.path.basename(log_file)
    # 使用正则表达式提取style名称
    style_match = re.search(r'gpt-4o-mini_SorryBench03_baseline_(.+?)\.log', filename)
    if style_match:
        style = style_match.group(1)
    else:
        style = "unknown"
    
    print(f"处理文件: {log_file}, style: {style}")
    
    # 初始化该style的统计数据
    if style not in scores_by_style:
        scores_by_style[style] = 0.0
        count_by_style[style] = 0
    
    # 读取日志文件
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 查找所有包含"StrongREJECT score"的行
        for line in lines:
            if "StrongREJECT score" in line:
                # 提取分数
                match = re.search(r'StrongREJECT score \d+: ([\d\.]+)', line)
                if match:
                    score = float(match.group(1))
                    scores_by_style[style] += score
                    count_by_style[style] += 1
                    print(f"  找到分数: {score}")
        
        # 检查是否有平均分数行
        for line in lines:
            if "Average StrongREJECT Score:" in line:
                match = re.search(r'Average StrongREJECT Score: ([\d\.]+)', line)
                if match:
                    avg_score = float(match.group(1))
                    # 如果还没有找到个别分数，使用平均分数
                    if count_by_style[style] == 0:
                        scores_by_style[style] = avg_score
                        count_by_style[style] = 1
                        print(f"  使用平均分数: {avg_score}")
        
    except Exception as e:
        print(f"处理文件 {log_file} 时出错: {e}")

# 计算每个style的平均分数
avg_scores = {}
for style in scores_by_style:
    if count_by_style[style] > 0:
        avg_scores[style] = scores_by_style[style] / count_by_style[style]
    else:
        print(f"警告: {style} 没有找到分数")
        avg_scores[style] = 0

# 按分数从大到小排序
sorted_styles = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

# 打印结果
print("\n所有prompt_style的StrongREJECT score均值排名（从大到小）：")
print("-" * 70)
print(f"{'排名':<6}{'Prompt Style':<30}{'平均分数':<12}{'样本数':<10}")
print("-" * 70)

for rank, (style, score) in enumerate(sorted_styles, 1):
    print(f"{rank:<6}{style:<30}{score:.6f}{count_by_style[style]:<10}") 