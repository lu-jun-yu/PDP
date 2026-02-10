#!/usr/bin/env python3
"""
Resource Setup Script
自动从 token.json 读取配置并执行模型和数据集下载
"""

import json
import subprocess
import sys
import os
from pathlib import Path


def load_tokens(token_file='token.json'):
    """从 token.json 文件加载配置"""
    try:
        with open(token_file, 'r', encoding='utf-8') as f:
            tokens = json.load(f)
        return tokens
    except FileNotFoundError:
        print(f"错误: 找不到 {token_file} 文件")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误: {token_file} 文件格式不正确")
        sys.exit(1)


def execute_command(command, description=""):
    """执行单个命令并实时显示输出"""
    if description:
        print(f"\n{'='*60}")
        print(f"执行: {description}")
        print(f"{'='*60}")

    print(f"命令: {command}")
    print()  # 空行，让输出更清晰

    try:
        # 不捕获输出，让命令的输出直接显示到终端
        # 这样可以看到实时进度（如下载进度条）
        result = subprocess.run(
            command,
            shell=True,
            check=True
        )

        print(f"\n✓ 完成: {description if description else command}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ 错误: 命令执行失败")
        print(f"返回码: {e.returncode}")
        return False


def main():
    """主函数"""
    print("="*60)
    print("资源准备脚本 - Resource Setup")
    print("="*60)

    # 加载配置
    print("\n[1/6] 加载配置文件...")
    tokens = load_tokens()
    WANDB_API_KEY = tokens.get('WANDB_API_KEY', '')
    hf_token = tokens.get('hf_token', '')

    if not WANDB_API_KEY or not hf_token:
        print("错误: token.json 中缺少必要的配置项")
        sys.exit(1)

    print(f"✓ 配置加载成功")
    print(f"  - WANDB_API_KEY: {WANDB_API_KEY[:20]}...")
    print(f"  - hf_token: {hf_token[:20]}...")

    # 定义要执行的命令列表
    commands = [
        # {
        #     'cmd': 'export HF_ENDPOINT=http://192.168.50.202:18090 >> ~/.bashrc',
        #     'desc': '设置 HuggingFace 镜像端点'
        # },
        # {
        #     'cmd': f'export WANDB_API_KEY={WANDB_API_KEY} >> ~/.bashrc',
        #     'desc': '设置 Weights & Biases API Key'
        # },
        # {
        #     'cmd': f"huggingface-cli download Qwen/Qwen3-0.6B --local-dir /root/LawShiftLLM/models/Qwen3-0.6B",
        #     'desc': '下载 Qwen3-0.6B 模型'
        # },
        {
            'cmd': f"huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /root/LawShiftLLM/models/Qwen2.5-7B-Instruct",
            'desc': '下载 Qwen2.5-7B-Instruct 模型'
        }
    ]

    # 逐个执行命令
    success_count = 0
    failed_count = 0

    for i, cmd_info in enumerate(commands, start=2):
        print(f"\n[{i}/6] {cmd_info['desc']}")

        if execute_command(cmd_info['cmd'], cmd_info['desc']):
            success_count += 1
        else:
            failed_count += 1
            response = input("\n命令执行失败，是否继续? (y/n): ")
            if response.lower() != 'y':
                print("用户中止执行")
                break

    # 总结
    print("\n" + "="*60)
    print("执行完成")
    print("="*60)
    print(f"成功: {success_count} 个命令")
    print(f"失败: {failed_count} 个命令")
    print("="*60)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
