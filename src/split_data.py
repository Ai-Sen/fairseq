import random

# 获取所有txt文件的文件名
txt_files = ["file1.txt", "file2.txt", "file3.txt"]  # 将文件名替换为您实际的文件名

# 合并所有txt文件
combined_data = []

for file_name in txt_files:
    with open(file_name, 'r', encoding='utf-8') as file:
        file_data = file.read()
        combined_data.append(file_data)

# 用空行分隔文档并合并成一个字符串
combined_text = "\n\n".join(combined_data)

# 随机打乱文档顺序
combined_lines = combined_text.split('\n\n')
random.shuffle(combined_lines)

# 计算分割点
total_docs = len(combined_lines)
train_split = int(0.7 * total_docs)
test_split = int(0.15 * total_docs)

# 分割数据集
train_data = "\n\n".join(combined_lines[:train_split])
test_data = "\n\n".join(combined_lines[train_split:train_split + test_split])
valid_data = "\n\n".join(combined_lines[train_split + test_split:])

# 写入分割后的文件
with open('data.train.raw', 'w', encoding='utf-8') as train_file:
    train_file.write(train_data)

with open('data.test.raw', 'w', encoding='utf-8') as test_file:
    test_file.write(test_data)

with open('data.valid.raw', 'w', encoding='utf-8') as valid_file:
    valid_file.write(valid_data)
