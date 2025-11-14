import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import numpy as np

# 定义输入JSON文件路径
json_files = ['ZH_EN_Hits1.json', 'JA_EN_Hits1.json', 'FR_EN_Hits1.json']
titles = ['DBP15K zh-en Hits1', 'DBP15K ja-en Hits1', 'DBP15K fr-en Hits1']


# 函数用于生成单个图像
def plot_mrr(json_file_path, title):
    # 读取json文件
    with open(json_file_path, 'r') as json_file:
        mrr_data = json.load(json_file)

    # 创建绘图区域
    fig, ax = plt.subplots(figsize=(5 / 2.54, 3.5 / 2.54), dpi=600)  # 5厘米宽，600dpi
    canvas = FigureCanvas(fig)

    # 遍历json中的每个txt文件的MRR数据并绘制曲线
    for file_name, mrr_list in mrr_data.items():
        ax.plot(mrr_list, label=file_name, linewidth=0.75)

    # 设置图表标题和标签
    ax.set_xlabel('Epoch', fontsize=6,fontname='Times New Roman')
    ax.set_ylabel(title, fontsize=6,fontname='Times New Roman')

    # 设置刻度线
    ax.tick_params(axis='both', direction='in', length=1, width=0.75, colors='black', labelsize=4)

    # 固定图例在右下角，字体为4pt
    ax.legend(loc='lower right', fontsize=6,framealpha=0.3)

    # 设置网格
    ax.grid(False)

    # 调整布局
    plt.tight_layout()

    # 保存图像到内存
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    plt.close(fig)
    return image


# 生成三个图像
images = [plot_mrr(json_file, title) for json_file, title in zip(json_files, titles)]

# 将三个图像拼接成一个图像，使用最小的间距
final_image = np.concatenate(images, axis=1)

# 将拼接后的图像转换为PIL格式并保存
final_image_pil = Image.fromarray(final_image)
final_image_pil.save('MRR.tif', format='TIFF')
final_image_pil.show()  # 可以显示拼接后的图像
