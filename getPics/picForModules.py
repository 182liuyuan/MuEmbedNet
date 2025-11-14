import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 数据
data = {
    "模型": ["A1", "B1", "A0+B1", "A1+B0", "A1+B1"],
    "DBP_ZH-EN_Hit@1": [0.821, 0.704, 0.783, 0.802, 0.844],
    "DBP_ZH-EN_Hit@10": [0.936, 0.837, 0.916, 0.898, 0.954],
    "DBP_ZH-EN_MRR": [0.862, 0.750, 0.831, 0.836, 0.883],
    "DBP_JA-EN_Hit@1": [0.848, 0.773, 0.821, 0.845, 0.872],
    "DBP_JA-EN_Hit@10": [0.951, 0.888, 0.935, 0.937, 0.971],
    "DBP_JA-EN_MRR": [0.885, 0.813, 0.863, 0.880, 0.909],
    "DBP_FR-EN_Hit@1": [0.941, 0.905, 0.935, 0.953, 0.962],
    "DBP_FR-EN_Hit@10": [0.979, 0.959, 0.986, 0.980, 0.992],
    "DBP_FR-EN_MRR": [0.955, 0.924, 0.953, 0.964, 0.972]
}

df = pd.DataFrame(data)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义数据集名称和指标
datasets = ['DBP_ZH-EN', 'DBP_JA-EN', 'DBP_FR-EN']
metrics = ['Hit@1', 'Hit@10', 'MRR']

# 创建自定义颜色映射
colors = [
    '#440154',  # 深紫色
    '#3b528b',  # 深蓝色
    '#21908d',  # 蓝绿色
    '#5ec962',  # 绿色
    '#fde725',  # 黄色
    '#ff7f0e',  # 橙色
    '#d62728',  # 红色
]

# 为每个数据集生成独立的图表
for dataset in datasets:
    plt.figure(figsize=(12, 8))

    # 设置柱子的宽度和位置
    bar_width = 0.2
    index = np.arange(len(df['模型']))

    # 绘制每个指标的柱状图
    for i, metric in enumerate(metrics):
        plt.bar(index + i * bar_width, df[f'{dataset}_{metric}'], bar_width, label=metric, color=colors[i])

    # 将标题放置在图表的正上方中间位置
    plt.figtext(0.5, 0.88, f'{dataset}', fontsize=24, ha='center')  # 调整标题位置

    plt.xticks(index + bar_width, df['模型'], fontsize=15)

    # 将图例横着放置在图表上方右侧
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.1), fontsize=16, ncol=3)

    # 添加数值标签
    for i, metric in enumerate(metrics):
        for j, value in enumerate(df[f'{dataset}_{metric}']):
            plt.text(j + i * bar_width, value + 0.01, f'{value:.3f}', ha='center', fontsize=15)

    # 去掉x轴和y轴的标签
    plt.xlabel('')
    plt.ylabel('')

    # 调整布局，增加顶部间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)  # 调整顶部间距

    # 保存图表
    plt.savefig(f'ablation_study_{dataset}.png', dpi=600, bbox_inches='tight')

    # 显示图表
    plt.show()