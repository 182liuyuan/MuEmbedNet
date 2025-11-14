import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 数据
set_size = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
precision = [0.2115, 0.391, 0.45769, 0.57628, 0.66987, 0.751923, 0.773, 0.773, 0.7576923, 0.748077]
recall = [0.2115, 0.4128, 0.4641, 0.602564, 0.690384, 0.771153, 0.774, 0.7735, 0.76987, 0.7480769]
f1 = [0.2115, 0.40256, 0.4589, 0.58397, 0.682692, 0.7608974, 0.7731, 0.773, 0.76987, 0.7480765]

# 创建图表，尺寸为13 cm x 8 cm
plt.figure(figsize=(13/2.54, 8/2.54))  # 将厘米转为英寸

# 绘制更细的折线图，使用不同的空心标记样式：空心圆圈、三角形、正方形
plt.plot(set_size, precision, 'o-', label="Precision", color='blue', marker='o', markersize=6, markerfacecolor='none', linewidth=1)
plt.plot(set_size, recall, '^-', label="Recall", color='green', marker='^', markersize=6, markerfacecolor='none', linewidth=1)
plt.plot(set_size, f1, 's-', label="F1-score", color='red', marker='s', markersize=6, markerfacecolor='none', linewidth=1)

# 添加标签、标题和图例
plt.xlabel('Training set sizes (%)', fontsize=8, fontname='Times New Roman')
plt.ylabel('Average performance', fontsize=8, fontname='Times New Roman')
plt.xticks(set_size, fontsize=6, fontname='Times New Roman')  # 保证横坐标为指定值
plt.yticks(np.arange(0.2, 0.91, 0.1), fontsize=6, fontname='Times New Roman')
plt.ylim(0.2, 0.9)

# 设置横坐标范围为从10开始
plt.xlim(10, 100)

# 移除背景方格
plt.grid(False)

# 将刻度线朝向正轴
plt.tick_params(axis='both', direction='in', length=4, width=0.8, colors='black')

# 设置图例，字体大小为6pt
plt.legend(loc="lower right", prop={'size': 6, 'family': 'Times New Roman'})

# 保存图像时设置像素大小，确保宽度为2047像素（13cm），同时DPI为400
plt.tight_layout()
plt.savefig('precision_recall_f1_plot_fixed_x101.png', dpi=400)

# 用 Pillow 打开 RGB 图像并检查尺寸
image = Image.open('../pics/precision_recall_f1_plot_fixed_x101.png')
print(f"Original image size: {image.size} (pixels)")

# 转换为 CMYK 模式，并检查转换后的尺寸
cmyk_image = image.convert('CMYK')
print(f"CMYK image size: {cmyk_image.size} (pixels)")

# 如果尺寸变化，调整为原始尺寸
if cmyk_image.size != image.size:
    cmyk_image = cmyk_image.resize(image.size, Image.ANTIALIAS)

# 再次检查尺寸
print(f"Resized CMYK image size: {cmyk_image.size} (pixels)")

# 保存为 CMYK 模式的 JPEG 图像，使用高质量参数
cmyk_image.save('precision_recall_f1_plot_fixed_x101.jpg', quality=95, dpi=(400, 400))
plt.show()
