#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试导入功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("测试导入 stock_market_prediction 模块...")

try:
    # 测试直接导入
    import stock_market_prediction
    print("✓ 成功导入 stock_market_prediction 包")
    
    # 测试导入 utils 模块
    from stock_market_prediction import utils
    print("✓ 成功导入 stock_market_prediction.utils 模块")
    
    # 测试导入具体函数
    from stock_market_prediction.utils import create_gui, fetch_market_data
    print("✓ 成功导入 utils 模块中的函数")
    
    # 测试导入 main 模块
    from stock_market_prediction import main
    print("✓ 成功导入 stock_market_prediction.main 模块")
    
    print("\n所有导入测试通过！模块结构正确。")
    
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("请确保项目已正确安装: pip install -e .")