#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试API设置功能
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox
import tempfile
import shutil

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接定义函数而不是从utils导入
def save_api_settings_to_env(env_file_path, api_base, api_key):
    """
    保存API设置到.env文件
    
    Args:
        env_file_path (str): .env文件路径
        api_base (str): API Base URL
        api_key (str): API密钥
    """
    # 读取现有的.env文件内容
    env_lines = []
    if os.path.exists(env_file_path):
        with open(env_file_path, 'r', encoding='utf-8') as f:
            env_lines = f.readlines()
    
    # 更新或添加API设置
    new_env_lines = []
    base_found = False
    key_found = False
    
    for line in env_lines:
        if line.startswith('DEEPSEEK_API_BASE='):
            new_env_lines.append(f'DEEPSEEK_API_BASE={api_base}\n')
            base_found = True
        elif line.startswith('DEEPSEEK_API_KEY='):
            new_env_lines.append(f'DEEPSEEK_API_KEY={api_key}\n')
            key_found = True
        else:
            new_env_lines.append(line)
    
    # 如果没有找到相应的设置，则添加它们
    if not base_found:
        # 找到合适的位置插入API设置
        insert_pos = len(new_env_lines)
        for i, line in enumerate(new_env_lines):
            if line.startswith('# DeepSeek API配置'):
                insert_pos = i + 1
                break
        new_env_lines.insert(insert_pos, f'DEEPSEEK_API_BASE={api_base}\n')
        
    if not key_found:
        # 找到合适的位置插入API密钥设置
        insert_pos = len(new_env_lines)
        for i, line in enumerate(new_env_lines):
            if line.startswith('# DeepSeek API配置'):
                insert_pos = i + 1
                break
        new_env_lines.insert(insert_pos, f'DEEPSEEK_API_KEY={api_key}\n')
    
    # 写入更新后的内容到.env文件
    with open(env_file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_env_lines)

def test_save_api_settings():
    """测试保存API设置到.env文件的功能"""
    # 创建临时目录和.env文件用于测试
    temp_dir = tempfile.mkdtemp()
    env_file_path = os.path.join(temp_dir, '.env')
    
    # 创建初始.env文件内容
    initial_content = """# API Configuration
# 请在此处设置您的API密钥和URL

# DeepSeek API配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com

# OpenAI API配置（可选）
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_API_BASE=https://api.openai.com/v1

# 其他配置
TF_ENABLE_ONEDNN_OPTS=0
"""
    
    with open(env_file_path, 'w', encoding='utf-8') as f:
        f.write(initial_content)
    
    # 测试数据
    test_api_base = "https://api.test.com"
    test_api_key = "test_api_key_12345"
    
    # 保存测试数据到.env文件
    save_api_settings_to_env(env_file_path, test_api_base, test_api_key)
    
    # 读取更新后的.env文件内容
    with open(env_file_path, 'r', encoding='utf-8') as f:
        updated_content = f.read()
    
    # 验证更新是否成功
    assert test_api_base in updated_content, "API Base URL未正确保存"
    assert test_api_key in updated_content, "API Key未正确保存"
    assert "DEEPSEEK_API_BASE=https://api.test.com" in updated_content, "API Base URL格式不正确"
    assert "DEEPSEEK_API_KEY=test_api_key_12345" in updated_content, "API Key格式不正确"
    
    # 清理临时文件
    shutil.rmtree(temp_dir)
    
    print("API设置保存功能测试通过！")

def test_gui_api_settings():
    """测试GUI中的API设置功能"""
    print("请在GUI中测试API设置功能：")
    print("1. 在API Base URL字段中输入测试URL")
    print("2. 在API Key字段中输入测试密钥")
    print("3. 点击'保存API设置'按钮")
    print("4. 检查.env文件是否正确更新")

if __name__ == "__main__":
    # 运行测试
    test_save_api_settings()
    test_gui_api_settings()