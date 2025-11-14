import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据
data = {
    "模型": ["MTransE", "JAPE", "IPTransE", "AlignEA", "BootEA", "RSN", "SEA", "MuGCN", "TransEdge", "AliNet", "RREA",
             "MRAEA", "Dual-AMN", "MSNEA", "LightEA", "BNGNN", "DSEA", "PSNEA", "MCEA", "CAEA", "GRGCN", "DvGNet",
             "MEEA", "SPSEA", "MuEmbedNet"],
    "DBPZH-EN_Hit@1": [0.308, 0.412, 0.543, 0.472, 0.629, 0.508, 0.424, 0.494, 0.753, 0.539, 0.715, 0.757, 0.808, 0.601,
                       0.812, 0.792, 0.789, 0.816, 0.821, 0.603, 0.457, 0.534, 0.768, 0.821, 0.844],
    "DBPZH-EN_Hit@10": [0.614, 0.745, 0.791, 0.792, 0.848, 0.745, 0.796, 0.844, 0.919, 0.826, 0.929, 0.930, 0.940,
                        0.830, 0.915, 0.949, 0.892, 0.957, "-", 0.788, 0.759, 0.844, 0.896, 0.943, 0.954],
    "DBPZH-EN_MRR": [0.364, 0.490, 0.578, 0.581, 0.703, 0.591, 0.548, 0.611, 0.801, 0.628, 0.794, 0.827, 0.857, 0.684,
                     0.849, 0.851, 0.819, 0.869, "-", 0.755, 0.684, 0.638, 0.835, 0.859, 0.883],
    "DBPJA-EN_Hit@1": [0.279, 0.363, 0.552, 0.448, 0.622, 0.507, 0.385, 0.501, 0.719, 0.549, 0.713, 0.758, 0.801, 0.535,
                       0.821, 0.779, 0.791, 0.819, 0.813, 0.613, 0.468, 0.538, 0.753, 0.817, 0.872],
    "DBPJA-EN_Hit@10": [0.575, 0.685, 0.798, 0.789, 0.854, 0.737, 0.783, 0.857, 0.932, 0.831, 0.933, 0.934, 0.949,
                        0.775, 0.933, 0.955, 0.894, 0.963, "-", 0.886, 0.771, 0.863, 0.909, 0.938, 0.971],
    "DBPJA-EN_MRR": [0.595, 0.476, 0.593, 0.563, 0.701, 0.590, 0.518, 0.621, 0.795, 0.645, 0.793, 0.826, 0.855, 0.617,
                     0.864, 0.844, 0.826, 0.853, "-", 0.691, 0.658, 0.651, 0.854, 0.867, 0.909],
    "DBPFR-EN_Hit@1": [0.244, 0.324, 0.578, 0.481, 0.653, 0.516, 0.400, 0.495, 0.710, 0.522, 0.739, 0.781, 0.840, 0.543,
                       0.863, 0.793, 0.847, 0.844, 0.850, 0.561, 0.472, 0.557, 0.783, 0.852, 0.962],
    "DBPFR-EN_Hit@10": [0.556, 0.667, 0.822, 0.824, 0.874, 0.768, 0.797, 0.870, 0.941, 0.852, 0.946, 0.948, 0.965,
                        0.801, 0.959, 0.966, 0.951, 0.982, "-", 0.867, 0.761, 0.881, 0.912, 0.949, 0.992],
    "DBPFR-EN_MRR": [0.335, 0.430, 0.609, 0.599, 0.731, 0.605, 0.533, 0.621, 0.796, 0.657, 0.816, 0.849, 0.888, 0.630,
                     0.900, 0.858, 0.883, 0.891, "-", 0.711, 0.675, 0.668, 0.863, 0.867, 0.972]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 将缺失值处理为0
df.replace('-', 0, inplace=True)

# 将字符串类型的0转换为浮点数
for col in df.columns[1:]:
    df[col] = df[col].astype(float)

# 计算每个模型在三个数据集上的平均值
df['Avg_Hit@1'] = df[['DBPZH-EN_Hit@1', 'DBPJA-EN_Hit@1', 'DBPFR-EN_Hit@1']].mean(axis=1)
df['Avg_Hit@10'] = df[['DBPZH-EN_Hit@10', 'DBPJA-EN_Hit@10', 'DBPFR-EN_Hit@10']].mean(axis=1)
df['Avg_MRR'] = df[['DBPZH-EN_MRR', 'DBPJA-EN_MRR', 'DBPFR-EN_MRR']].mean(axis=1)

# 设置绘图风格
sns.set_theme(style="whitegrid", palette="viridis")

# 为每个指标生成独立的横向柱状图
metrics = ['Avg_Hit@1', 'Avg_Hit@10', 'Avg_MRR']
titles = ['Avg Hit@1', 'Avg Hit@10', 'Avg MRR']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    plt.figure(figsize=(15, 8))

    # 将 MuEmbedNet 放到最后
    sorted_df = df.sort_values(by=metric, ascending=True)
    mu_embed_row = sorted_df[sorted_df['模型'] == 'MuEmbedNet']
    other_rows = sorted_df[sorted_df['模型'] != 'MuEmbedNet']
    sorted_df = pd.concat([other_rows, mu_embed_row])

    ax = sns.barplot(x='模型', y=metric, data=sorted_df, palette='viridis')

    # 设置 y 轴标签为标题内容
    ax.set_ylabel(title, fontsize=16)
    ax.set_xlabel('模型', fontsize=16)

    # 设置x轴标签旋转角度
    plt.xticks(rotation=45, ha='right', fontsize=16)

    # 高亮 MuEmbedNet
    mu_embed_idx = sorted_df['模型'].tolist().index('MuEmbedNet')
    ax.patches[mu_embed_idx].set_color('red')

    # 添加数值标签
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2, height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=14)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    plt.savefig(f'model_comparison_{metric}_horizontal.png', dpi=600, bbox_inches='tight')
    plt.show()