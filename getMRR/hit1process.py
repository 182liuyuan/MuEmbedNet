import os
import re

# 文件夹路径
# folder_path = "ja_en_output"
# outputfile = "JA_Hits1"

folder_path = "../EA_Data/GeoOutput"
outputfile = "../EA_Data/GeoOutput/hits_MRR"

if not os.path.exists(outputfile):
    os.makedirs(outputfile)

# 正则表达式模式
# epoch_pattern = re.compile(r"Epoch:\s*(\d+)\s*/\s*\d+.*?Epoch:\s*(\d+)\s*/\s*\d+", re.DOTALL)

epoch_pattern = re.compile(r"Epoch:\s*(\d+)\s*/\s*\d+")

test_set_pattern = re.compile(
    r"----------------Test_set-----------------.*?Left:\s*Hits@1:\s*(\d+\.\d+)%.*?Right:\s*Hits@1:\s*(\d+\.\d+)%", re.DOTALL
)

# 遍历文件夹下所有txt文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        print(f"Processing file: {file_name}")
        input_file_path = os.path.join(folder_path, file_name)
        output_file_path = os.path.join(outputfile, f"{os.path.splitext(file_name)[0]}_Hits1.txt")

        with open(input_file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # 提取每个区块的 Epoch 和对应的 Test_set Hits@1 数据
        epoch_matches = epoch_pattern.findall(content)
        test_set_matches = test_set_pattern.findall(content)

        if len(epoch_matches) == len(test_set_matches):  # 检查Epoch和Test_set的数量是否一致
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                for i, match in enumerate(test_set_matches):
                    # 获取每个区块的第二个 Epoch
                    epoch = epoch_matches[i]
                    left_hit = float(match[0]) / 100
                    right_hit = float(match[1]) / 100
                    average_hit = (left_hit + right_hit) / 2

                    # 写入输出文件
                    output_file.write(
                        f"Epoch {epoch}: Hits@1 Left = {left_hit:.3f}, Hits@1 Right = {right_hit:.3f}, Average Hits@1 = {average_hit:.3f}\n"
                    )
            print(f"File {file_name}: Successfully processed and saved to {output_file_path}.")
        else:
            print(len(epoch_matches),len(test_set_matches))
            print(f"File {file_name}: Mismatch in data lengths. Skipping.")

print("处理完成，所有结果已保存！")
