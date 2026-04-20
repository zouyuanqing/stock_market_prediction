"""
Stock Market Prediction - LLM Agent 版本

一个由LLM驱动的智能股票分析系统，可以自主选择工具、决定模型参数并进行多步推理。

## 特性

- 🤖 **LLM自主决策**: Agent可以自主选择使用哪些工具
- 🎯 **智能参数选择**: LLM决定ARIMA和LSTM的超参数
- 🔍 **网络搜索**: 自动搜索相关新闻和市场动态
- 📊 **技术分析**: 完整的指标系统 (MA, EMA, MACD, RSI, 布林带)
- 🧠 **ReAct范式**: Reasoning + Acting的多步推理

## 来源声明

本项目的设计灵感来自:
- [OpenCLAW](https://github.com/OpenCLAW/OpenCLAW) - Apache 2.0 License
- [Hermes Agent](https://github.com/NousResearch/Hermes-Function-Calling) - MIT License

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 配置API Key

在 `.env` 文件中设置:
```
DEEPSEEK_API_KEY=your_api_key_here
```

### 2. 运行分析

```python
from stock_market_prediction.agents.llm_agent import StockAnalysisAgent

# 创建Agent
agent = StockAnalysisAgent(api_key="your_api_key")

# 分析股票
results = agent.analyze("AAPL")

# 查看结果
print(results["final_analysis"])
```

### 3. 命令行使用

```bash
python -m stock_market_prediction.agents.llm_agent
```

## 可用工具

| 工具名 | 描述 |
|--------|------|
| `stock_data_fetcher` | 获取股票历史数据 |
| `web_search` | 搜索网络信息 |
| `arima_predictor` | ARIMA时间序列预测 |
| `lstm_predictor` | LSTM深度学习预测 |
| `technical_analyzer` | 技术指标计算 |

## LLM参数选择指南

### ARIMA参数
- **p** (AR阶数): 1-3，趋势明显用高值
- **d** (差分阶数): 通常1
- **q** (MA阶数): 1-2

### LSTM参数
- **hidden_size**: 32-128
- **num_layers**: 1-3
- **lookback_days**: 30-60
- **epochs**: 30-100

## 示例输出

Agent会自动执行以下步骤:
1. 获取股票历史数据
2. 搜索相关新闻
3. 计算技术指标
4. 运行ARIMA和LSTM预测
5. 综合分析给出建议

## 项目结构

```
stock_market_prediction/
├── agents/
│   └── llm_agent.py      # LLM驱动的Agent
├── tools/
│   └── __init__.py       # 工具模块
├── utils.py              # 传统工具函数 (向后兼容)
└── main.py               # 主入口
```

## License

MIT License
"""

__version__ = "3.0.0"
__author__ = "Stock Analysis Team"
