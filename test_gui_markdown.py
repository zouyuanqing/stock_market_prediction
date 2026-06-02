#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试GUI中的markdown渲染功能
"""

import sys
import os
import tkinter as tk
from tkinter import ttk

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_market_prediction.utils import convert_markdown_to_text, format_text_widget

def test_gui_markdown_rendering():
    """测试GUI中的markdown渲染功能"""
    
    # 创建测试窗口
    root = tk.Tk()
    root.title("Markdown渲染测试")
    root.geometry("600x400")
    
    # 创建文本控件
    text_widget = tk.Text(root, wrap=tk.WORD)
    text_widget.pack(fill='both', expand=True, padx=10, pady=10)
    
    # 测试markdown文本
    test_markdown = """
# 股票市场分析报告
本次分析基于最新的市场数据。

## 技术分析
- **趋势**：当前市场呈上升趋势
- **支撑位**：关键支撑位在150.00
- **阻力位**：初步阻力位在165.00

### 技术指标
1. RSI(14)：55 - 接近超买
2. MACD：正值，呈现金叉
3. 布林带：价格接近上轨

## 投资建议
__谨慎乐观__：建议逢低买入，设置止损位。
"""
    
    # 转换并显示markdown
    formatted_text = convert_markdown_to_text(test_markdown)
    text_widget.insert(tk.END, formatted_text)
    
    # 应用格式化
    format_text_widget(text_widget)
    
    # 设置文本为只读
    text_widget.config(state=tk.DISABLED)
    
    # 添加滚动条
    scrollbar = tk.Scrollbar(text_widget, orient="vertical", command=text_widget.yview)
    scrollbar.pack(side="right", fill="y")
    text_widget.config(yscrollcommand=scrollbar.set)
    
    print("GUI Markdown渲染测试窗口已创建，请查看GUI窗口")
    print("关闭窗口以结束测试")
    
    # 运行GUI
    root.mainloop()

if __name__ == "__main__":
    test_gui_markdown_rendering()