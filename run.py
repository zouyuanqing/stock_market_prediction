#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
直接运行脚本，避免模块导入问题
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 从utils模块导入create_gui函数
from stock_market_prediction.utils import create_gui

def main():
    create_gui()

if __name__ == "__main__":
    main()