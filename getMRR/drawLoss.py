# import json
# import matplotlib.pyplot as plt
# # 定义输入JSON文件路径
# # json_file_path = 'geo_losses.json'
# # json_file_path = 'ja_en_losses.json'
# json_file_path = 'fr_en_losses.json'
#
# # 读取JSON文件
# with open(json_file_path, 'r') as file:
#     data = json.load(file)
#
# # 遍历所有数据集，提取并绘制test_losses
# for dataset_name, losses in data.items():
#     test_losses = losses["test_losses"]
#     plt.plot(test_losses, label=f'{dataset_name} Test Losses')
#
# # 设置图表标题和标签
# plt.xlabel('Epoch')
# plt.ylabel('DBP15K zh-en test Losses')
# if json_file_path=='ja_en_losses.json':
#     plt.ylabel('DBP15K ja-en test Losses')
# elif json_file_path=='fr_en_losses.json':
#     plt.ylabel('DBP15K fr-en test Losses')
# plt.grid(False)
# # 将刻度线朝向正轴
# plt.tick_params(axis='both', direction='in', length=4, width=0.8, colors='black')
# # 设置图例，字体大小为6pt
# plt.legend(loc="upper right")
# # 保存图像时设置像素大小，确保宽度为2047像素（13cm），同时DPI为400
# plt.tight_layout()
# plt.savefig('precision_recall_f1_plot_fixed_x101.png', dpi=800)
# # 显示图表
# plt.show()

# import json
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from PIL import Image
# import numpy as np
#
# # 定义输入JSON文件路径
# json_files = ['geo_losses.json', 'ja_en_losses.json', 'fr_en_losses.json']
# titles = ['DBP15K zh-en test Losses', 'DBP15K ja-en test Losses', 'DBP15K fr-en test Losses']
#
#
# # 函数用于生成单个图像
# def plot_test_losses(json_file_path, title):
#     # 读取JSON文件
#     with open(json_file_path, 'r') as file:
#         data = json.load(file)
#
#     # 创建绘图区域
#     fig, ax = plt.subplots(figsize=(5 / 2.54, 3/ 2.54), dpi=600)  # 宽度为5厘米，600dpi
#     canvas = FigureCanvas(fig)
#
#     # 遍历所有数据集，提取并绘制test_losses
#     for dataset_name, losses in data.items():
#         test_losses = losses["test_losses"]
#         ax.plot(test_losses, label=f'{dataset_name} Test Losses', linewidth=0.75)
#
#     # 设置图表标题和标签
#     ax.set_xlabel('Epoch', fontsize=4)
#     ax.set_ylabel(title, fontsize=4)
#
#     # 设置刻度线朝向正轴
#     ax.tick_params(axis='both', direction='in', length=2, width=0.75, colors='black', labelsize=4)
#
#     # 设置图例，字体大小为4pt
#     ax.legend(loc="upper right", fontsize=4)
#
#     # 调整布局
#     plt.tight_layout()
#
#     # 保存图像到内存
#     canvas.draw()
#     width, height = fig.get_size_inches() * fig.get_dpi()
#     image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
#
#     plt.close(fig)
#     return image
#
#
# # 生成三个图像
# images = [plot_test_losses(json_file, title) for json_file, title in zip(json_files, titles)]
#
# # 将三个图像拼接成一个图像
# final_image = np.concatenate(images, axis=1)
#
# # 将拼接后的图像转换为PIL格式并保存
# final_image_pil = Image.fromarray(final_image)
# final_image_pil.show()  # 可以显示拼接后的图像
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import numpy as np

# 定义输入JSON文件路径
json_files = ['geo_losses.json', 'ja_en_losses.json', 'fr_en_losses.json']
titles = ['DBP15K zh-en test Losses', 'DBP15K ja-en test Losses', 'DBP15K fr-en test Losses']


# 函数用于生成单个图像
def plot_test_losses(json_file_path, title):
    # 读取JSON文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # 创建绘图区域
    fig, ax = plt.subplots(figsize=(5 / 2.54, 3.5 / 2.54), dpi=600)  # 宽度为5厘米，600dpi
    canvas = FigureCanvas(fig)

    # 遍历所有数据集，提取并绘制test_losses
    for dataset_name, losses in data.items():
        test_losses = losses["test_losses"]
        ax.plot(test_losses, label=f'{dataset_name} Test Losses', linewidth=0.75)

    # 设置图表标题和标签
    ax.set_xlabel('Epoch', fontsize=4)
    ax.set_ylabel(title, fontsize=4)

    # 设置刻度线朝向正轴
    ax.tick_params(axis='both', direction='in', length=1, width=0.75, colors='black', labelsize=4)

    # 设置图例，字体大小为4pt
    ax.legend(loc="upper right", fontsize=4)

    # 调整布局
    plt.tight_layout()

    # 保存图像到内存
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    plt.close(fig)
    return image


# 生成三个图像
images = [plot_test_losses(json_file, title) for json_file, title in zip(json_files, titles)]

# 将三个图像拼接成一个图像，使用最小的间距
final_image = np.concatenate(images, axis=1)

# 将拼接后的图像转换为PIL格式并保存
final_image_pil = Image.fromarray(final_image)
final_image_pil.save('Losses.tif', format='TIFF')
final_image_pil.show()  # 可以显示拼接后的图像
