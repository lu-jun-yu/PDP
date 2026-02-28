"""
数据处理脚本：替换法律条文引用中的数字前缀

功能：
1. 处理 data/pku_fabao 下的 train、test 文件夹中的起诉、不起诉文件夹中的所有文件
2. 将"《{数字}中华人民共和国刑事诉讼法》"替换为《中华人民共和国刑事诉讼法》
3. 将"《{数字}中华人民共和国刑法》"替换为《中华人民共和国刑法》
4. 将"《{数字}人民检察院刑事诉讼规则》"替换为《人民检察院刑事诉讼规则》
"""

import os
import re
from pathlib import Path

# 尝试导入tqdm，如果没有则使用简单的进度显示
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc="", leave=False):
        return iterable

# ================= 配置区域 =================

# 数据根目录
BASE_DIR = "data/pku_fabao"

# 需要处理的文件夹
FOLDERS = ["train", "test"]

# 需要处理的子文件夹
SUBFOLDERS = ["起诉", "不起诉"]

# ===========================================

def replace_law_references(content):
    """
    替换法律条文引用中的数字前缀
    
    Args:
        content: 文件内容字符串
        
    Returns:
        替换后的内容字符串
    """
    # 替换《{数字}中华人民共和国刑事诉讼法》为《中华人民共和国刑事诉讼法》
    content = re.sub(r'《\d+中华人民共和国刑事诉讼法》', '《中华人民共和国刑事诉讼法》', content)
    
    # 替换《{数字}中华人民共和国刑法》为《中华人民共和国刑法》
    content = re.sub(r'《\d+中华人民共和国刑法》', '《中华人民共和国刑法》', content)

    # 替换《{数字}人民检察院刑事诉讼规则》为《人民检察院刑事诉讼规则》
    content = re.sub(r'《\d+人民检察院刑事诉讼规则》', '《人民检察院刑事诉讼规则》', content)
    
    return content

def process_file(file_path):
    """
    处理单个文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        (是否成功, 是否被修改)
    """
    try:
        # 读取文件
        with open(file_path, "r", encoding="utf-8") as f:
            original_content = f.read()
        
        # 执行替换
        new_content = replace_law_references(original_content)
        
        # 如果内容有变化，写回文件
        if new_content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True, True
        else:
            return True, False
            
    except Exception as e:
        print(f"\n处理文件失败 {file_path}: {e}")
        return False, False

def main():
    """主函数"""
    print("=" * 60)
    print("法律条文引用替换脚本")
    print("=" * 60)
    
    # 统计信息
    stats = {
        "total_files": 0,
        "processed_files": 0,
        "modified_files": 0,
        "failed_files": 0
    }
    
    # 检查基础目录是否存在
    if not os.path.exists(BASE_DIR):
        print(f"错误：基础目录 {BASE_DIR} 不存在，请检查路径。")
        return
    
    # 遍历所有文件夹
    for folder in FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        
        if not os.path.exists(folder_path):
            print(f"警告：文件夹 {folder_path} 不存在，跳过。")
            continue
        
        print(f"\n处理文件夹: {folder}")
        
        # 遍历起诉和不起诉子文件夹
        for subfolder in SUBFOLDERS:
            subfolder_path = os.path.join(folder_path, subfolder)
            
            if not os.path.exists(subfolder_path):
                print(f"  警告：子文件夹 {subfolder_path} 不存在，跳过。")
                continue
            
            # 获取所有txt文件
            txt_files = list(Path(subfolder_path).glob("*.txt"))
            
            if not txt_files:
                print(f"  子文件夹 {subfolder} 中没有找到txt文件，跳过。")
                continue
            
            print(f"  处理子文件夹: {subfolder} (共 {len(txt_files)} 个文件)")
            
            # 处理每个文件
            for file_path in tqdm(txt_files, desc=f"    {subfolder}", leave=False):
                stats["total_files"] += 1
                
                success, modified = process_file(str(file_path))
                
                if success:
                    stats["processed_files"] += 1
                    if modified:
                        stats["modified_files"] += 1
                else:
                    stats["failed_files"] += 1
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("处理完成！统计信息：")
    print("=" * 60)
    print(f"总文件数: {stats['total_files']}")
    print(f"成功处理: {stats['processed_files']}")
    print(f"已修改文件: {stats['modified_files']}")
    print(f"失败文件: {stats['failed_files']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
