# 定义文件路径
import os

file_path = "../baseline_output/baseline_ja_en_log.txt"
# output_file_path = "FR/"+file_path.split("/")[1].split(".")[0]+"_MRR.txt"
# file_path = "../EA_Data/GeoOutput/muEmbed_Geo_log.txt"
# # 下面这行是单独测试用的
output_file_path = "../baseline_output/"+file_path.split("/")[1].split(".")[0]+"_ja_en_MRR.txt"
# output_file_path = "../EA_Data/GeoOutput/"+file_path.split("/")[1].split(".")[0]+"_muEmbed__Geo_MRR.txt"
# 初始化一个字典来存储每个 epoch 的 MRR 值
epoch_mrr = {}

# 打开文件并读取内容
with open(file_path, 'r') as file:
    lines = file.readlines()

# 初始化行索引
line_index = 0

# 遍历文件的每一行
while line_index < len(lines):
    line = lines[line_index].strip()

    # 检查是否是Epoch行
    if line.startswith("Epoch:"):
        # 提取epoch数值
        epoch_number = int(line.split(':')[1].split('/')[0].strip())

    # 检查是否是Test_set中包含MRR的行
    if "----------------Test_set-----------------" in line:
        # 检查是否有足够的行来读取
        if line_index + 2 < len(lines):
            next_line_1 = lines[line_index + 1].strip()
            next_line_2 = lines[line_index + 2].strip()

            # 提取 left 和 right 的 MRR 值
            if "Left:" in next_line_1 and "MRR:" in next_line_1:
                mrr_left = float(next_line_1.split('MRR: ')[1])

            if "Right:" in next_line_2 and "MRR:" in next_line_2:
                mrr_right = float(next_line_2.split('MRR: ')[1])

            # 将 epoch 和 MRR 值存储到字典中
            epoch_mrr[epoch_number] = {'MRR Left': mrr_left, 'MRR Right': mrr_right}

        # 跳过读取的两行
        line_index += 2

    # 增加行索引
    line_index += 1
# 将提取的数据写入输出文件
with open(output_file_path, 'w') as output_file:
    for epoch, mrr_values in epoch_mrr.items():
        output_file.write(f"Epoch {epoch}: MRR Left = {mrr_values['MRR Left']:.3f}, MRR Right = {mrr_values['MRR Right']:.3f}\n")

# 输出每个 epoch 的 MRR 值
for epoch, mrr_values in epoch_mrr.items():
    print(f"Epoch {epoch}: MRR Left = {mrr_values['MRR Left']:.3f}, MRR Right = {mrr_values['MRR Right']:.3f}")
