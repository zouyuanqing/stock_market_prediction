#!/usr/bin/env python3
"""
主入口脚本 - 演示LLM Agent的股票分析能力
"""

import sys
import os
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_market_prediction.agents.llm_agent import StockAnalysisAgent


def main():
    """主函数"""
    print("=" * 70)
    print("📈 Stock Market Prediction with LLM Agent")
    print("=" * 70)
    
    # 获取API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("\n❌ 错误: 未找到 DEEPSEEK_API_KEY")
        print("\n请设置环境变量:")
        print("  export DEEPSEEK_API_KEY=your_api_key")
        print("\n或创建 .env 文件:")
        print("  cp .env.example .env")
        print("  然后编辑 .env 填入你的 API Key")
        sys.exit(1)
    
    # 解析命令行参数
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    
    # 计算日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    print(f"\n🎯 分析目标: {ticker}")
    print(f"📅 时间范围: {start_date} 至 {end_date}")
    print(f"⏱️  数据天数: {days} 天")
    
    # 创建Agent
    print("\n🤖 初始化 Agent...")
    agent = StockAnalysisAgent(api_key=api_key)
    
    print(f"\n✅ Agent已就绪，可用工具:")
    for tool in agent.available_tools:
        print(f"   • {tool['name']}: {tool['description']}")
    
    # 执行分析
    print("\n" + "=" * 70)
    results = agent.analyze(ticker, start_date=start_date, end_date=end_date)
    
    # 打印统计信息
    print("\n📊 分析统计:")
    print(f"   • 迭代次数: {results['iterations']}")
    print(f"   • 工具调用: {len(results['tool_calls'])} 次")
    
    if results['tool_calls']:
        print("\n   工具调用详情:")
        for call in results['tool_calls']:
            print(f"   [{call['iteration']}] {call['tool']}: {call['params']}")
    
    # 打印最终分析
    print("\n" + "=" * 70)
    print("📝 最终分析报告")
    print("=" * 70)
    print(results['final_analysis'])
    
    # 保存结果
    output_file = f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        print(f"\n💾 详细结果已保存到: {output_file}")
    except Exception as e:
        print(f"\n⚠️ 保存结果时出错: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 分析完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
