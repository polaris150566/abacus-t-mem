import os
import json
import shutil

def load_json(file_path):
    # 加载 JSON 文件并返回字典
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_files_to_copy(source_dir, prefix_list):
    # 获取指定目录中所有以 prefix_list 中元素为前缀的文件
    files_to_copy = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            for prefix in prefix_list:
                if file.startswith(prefix):
                    files_to_copy.append(os.path.join(root, file))
                    break  # 一旦找到匹配的前缀，跳出循环
    return files_to_copy

def copy_files(files, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for file in files:
        shutil.copy(file, destination_dir)
        print(f"已复制文件: {file}")

def main():
    # JSON 文件路径
    json_file = '/home/chenty/abacust_mem/src/data/data_storage/merged_cluster_dict.json'
    source_dir = '/home/chenty/abacust_mem/src/data/data_storage/assembled_pdbs'
    destination_dir = '/home/chenty/public_data/third_cluster_data/source_data'

    # 加载 JSON 文件
    data = load_json(json_file)

    # 获取 valid 键中的字典，汇总成一个大列表
    valid_dict = data.get('valid', {})
    combined_list = []
    for key, value in valid_dict.items():
        if isinstance(value, list):
            combined_list.extend(value)

    # 获取符合条件的文件
    files_to_copy = get_files_to_copy(source_dir, combined_list)

    # 将文件复制到目标目录
    copy_files(files_to_copy, destination_dir)

if __name__ == "__main__":
    main()