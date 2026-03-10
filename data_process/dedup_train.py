"""
从 train 集中删除与 test 集 ID 重复的样本。
ID 通过文件名中的 CLI.P.XXXXXXXX 模式匹配。
"""

import os
import re
import argparse

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "pku_fabao")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
ID_PATTERN = re.compile(r"CLI\.P\.\d+")


def extract_ids_from_dir(directory: str) -> set[str]:
    """递归收集目录下所有文件名中的 ID。"""
    ids = set()
    for root, _, files in os.walk(directory):
        for f in files:
            m = ID_PATTERN.search(f)
            if m:
                ids.add(m.group())
    return ids


def find_duplicates(train_dir: str, test_dir: str) -> list[str]:
    """返回 train 中与 test ID 重复的文件完整路径列表。"""
    test_ids = extract_ids_from_dir(test_dir)
    print(f"Test 集共 {len(test_ids)} 个唯一 ID")

    duplicates = []
    for root, _, files in os.walk(train_dir):
        for f in files:
            m = ID_PATTERN.search(f)
            if m and m.group() in test_ids:
                duplicates.append(os.path.join(root, f))
    return duplicates


def main():
    parser = argparse.ArgumentParser(description="删除 train 中与 test 重复的样本")
    parser.add_argument("--dry-run", action="store_true", help="仅列出重复文件，不删除")
    args = parser.parse_args()

    duplicates = find_duplicates(TRAIN_DIR, TEST_DIR)
    print(f"Train 集中发现 {len(duplicates)} 个与 Test 重复的文件:\n")

    for path in duplicates:
        print(f"  {os.path.relpath(path, DATA_ROOT)}")

    if not duplicates:
        print("无需删除。")
        return

    if args.dry_run:
        print(f"\n[dry-run] 以上 {len(duplicates)} 个文件未删除。去掉 --dry-run 执行实际删除。")
    else:
        for path in duplicates:
            os.remove(path)
        print(f"\n已删除 {len(duplicates)} 个重复文件。")


if __name__ == "__main__":
    main()
