#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试markdown渲染功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_market_prediction.utils import convert_markdown_to_text

def test_markdown_conversion():
    """测试markdown到文本的转换功能"""
    
    # 测试markdown文本
    test_markdown = """
# 数据概览
本次分析基于过去一年的股票数据。

## 趋势分析
- **上涨趋势**：股票价格整体呈上升趋势
- **波动性**：中期波动较大，需注意风险
- **支撑位**：关键支撑位在￥150附近

## 技术指标
1. **RSI**：目前处于55，接近超买区域
2. **MACD**：呈现金叉迹象
3. **布林带**：价格接近上轨

## 预测结果
**未来一周**：预计上涨2-3%
__风险提示__：需关注市场整体情绪变化
"""
    
    # 转换markdown到格式化文本
    formatted_text = convert_markdown_to_text(test_markdown)
    
    print("原始Markdown:")
    print(test_markdown)
    print("\n" + "="*50 + "\n")
    print("转换后的格式化文本:")
    print(formatted_text)
    
    # 验证关键元素是否正确转换
    assert "=== 数据概览 ===" in formatted_text
    assert "--- 趋势分析 ---" in formatted_text
    assert "• 上涨趋势" in formatted_text or "- *上涨趋势*" in formatted_text
    
    print("\n" + "="*50)
    print("Markdown转换测试通过！")

if __name__ == "__main__":
    test_markdown_conversion()