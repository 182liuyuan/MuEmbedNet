import os
import json

# 文件夹路径，替换为你的实际路径
folder_path = "FR"
output_json_path = "base_all/FR_EN_MRR.json"

# 初始化用于存储结果的字典
mrr_data = {}
# 遍历文件夹中的所有txt文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)

        # 读取文件内容
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 存储当前txt文件的Average MRR列表
        average_mrr_list = []

        # 处理每一行，提取Average MRR并添加到列表中
        for line in lines:
            if "Average MRR" in line:
                # 提取Average MRR的值
                average_mrr = float(line.split("Average MRR =")[1].strip())
                average_mrr_list.append(average_mrr)

        # 使用txt文件名作为键，将其对应的Average MRR列表保存到mrr_data字典
        mrr_data[file_name] = average_mrr_list

# 将mrr_data中的数据保存到JSON文件
with open(output_json_path, 'w') as json_file:
    json.dump(mrr_data, json_file, indent=4)

print(f"所有文件已处理完毕，结果已保存到 {output_json_path}")

