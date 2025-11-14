import os

# 文件夹路径，替换为你的实际路径
# folder_path = "ZH"  # 确保路径正确
folder_path = "../baseline_output"
# folder_path = "../EA_Data/GeoOutput"
# listdir =["FR/A1_A2_muView_fr_en_log_MRR.txt","JA/A1_A2_muView_ja_en_log_MRR.txt","ZH/A1_A2_muView_zh_en_log_MRR.txt"]
listdir =["../getMRR/FR/baseline_fr_en_MRR.txt","../getMRR/JA/baseline_ja_en_MRR.txt","../getMRR/ZH/baseline_zh_en_MRR.txt"]

# listdir =["../EA_Data/GeoOutput/EA_Data_baseline_Geo_MRR.txt","../EA_Data/GeoOutput/EA_Data_muEmbed_Geo_MRR.txt"]
# 遍历文件夹中的所有txt文件
for file_name in os.listdir(folder_path):
# for file_name in listdir:
    print(file_name)
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        # 读取文件内容
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 处理每一行，计算平均MRR并添加
        updated_lines = []
        for line in lines:
            if "MRR Left" in line and "MRR Right" in line:
                # 提取MRR Left和MRR Right的值
                mrr_left = float(line.split("MRR Left =")[1].split(",")[0].strip())
                mrr_right = float(line.split("MRR Right =")[1].strip())
                average_mrr = (mrr_left + mrr_right) / 2

                # 添加平均MRR到行末
                updated_line = line.strip() + f", Average MRR = {average_mrr:.3f}\n"
                print(updated_line)
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)

        # 将更新后的内容写回文件
        with open(file_path, 'w') as file:
            file.writelines(updated_lines)

print("所有文件已处理完毕。")
