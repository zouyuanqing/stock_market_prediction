# 📈 Stock Market Prediction with LLM Agent

一个由LLM驱动的智能股票分析系统，采用ReAct范式实现自主决策的股票分析Agent。

## ✨ 特性

- 🤖 **LLM自主决策**: Agent可以自主选择使用哪些工具
- 🎯 **智能参数选择**: LLM决定ARIMA和LSTM的超参数
- 🔍 **网络搜索**: 自动搜索相关新闻和市场动态  
- 📊 **技术分析**: 完整的指标系统 (MA, EMA, MACD, RSI, 布林带)
- 🧠 **ReAct范式**: Reasoning + Acting的多步推理
- 🔧 **可扩展工具**: 易于添加新工具

## 📋 来源声明

本项目的设计灵感来自以下开源项目:

- **[OpenCLAW](https://github.com/OpenCLAW/OpenCLAW)** - Apache 2.0 License
  - 参考了其工具调用架构设计
  
- **[Hermes Agent](https://github.com/NousResearch/Hermes-Function-Calling)** - MIT License
  - 参考了其函数调用格式和Prompt设计

我们感谢这些优秀项目的贡献！

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API Key

创建 `.env` 文件:

```bash
cp .env.example .env
```

编辑 `.env`:

```
DEEPSEEK_API_KEY=sk-your-api-key-here
```

### 3. 运行分析

#### Python API

```python
from stock_market_prediction.agents.llm_agent import StockAnalysisAgent

# 创建Agent
agent = StockAnalysisAgent(api_key="your_api_key")

# 分析股票
results = agent.analyze("AAPL", start_date="2023-01-01", end_date="2024-01-01")

# 查看结果
print(results["final_analysis"])
```

#### 命令行

```bash
python -m stock_market_prediction.agents.llm_agent
```

## 🛠️ 可用工具

| 工具名 | 描述 | LLM控制参数 |
|--------|------|-------------|
| `stock_data_fetcher` | 获取股票历史数据 | ticker, start_date, end_date |
| `web_search` | 搜索网络信息 | query, max_results |
| `arima_predictor` | ARIMA时间序列预测 | p, d, q, forecast_days |
| `lstm_predictor` | LSTM深度学习预测 | hidden_size, num_layers, lookback_days, epochs |
| `technical_analyzer` | 技术指标计算 | indicators列表 |

## 🎓 LLM参数选择指南

### ARIMA模型参数

LLM会根据数据特征自主选择:

- **p** (自回归阶数): 
  - 1: 基础模式
  - 2-3: 有明显趋势
  - >3: 复杂周期模式

- **d** (差分阶数):
  - 0: 数据已平稳
  - 1: 需要一阶差分 (最常用)
  - 2: 强烈趋势

- **q** (移动平均阶数):
  - 1: 基础噪声
  - 2-3: 复杂噪声结构

### LSTM模型参数

LLM会根据数据量和分析深度选择:

- **hidden_size**: 32/50/100/128
- **num_layers**: 1(简单)/2(标准)/3(深度)
- **lookback_days**: 30(短期)/60(中期)/90(长期)
- **epochs**: 30-100 (根据训练损失调整)

## 📁 项目结构

```
stock_market_prediction/
├── agents/
│   └── llm_agent.py      # LLM驱动的Agent核心
├── tools/
│   └── __init__.py       # 工具模块集合
├── utils.py              # 传统工具函数 (向后兼容)
├── main.py               # 主入口脚本
├── requirements.txt      # 依赖列表
├── .env.example          # 环境变量示例
└── README.md             # 本文档
```

## 🔬 工作原理

```
┌─────────────┐
│   用户请求   │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  LLM (DeepSeek) │ ←→ System Prompt (工具描述+规则)
└──────┬──────────┘
       │
       │ 输出: {"action": "tool_name", "action_input": {...}}
       ▼
┌─────────────────┐
│  工具执行器      │
└──────┬──────────┘
       │
       ├─→ stock_data_fetcher → Yahoo Finance
       ├─→ web_search → DuckDuckGo
       ├─→ arima_predictor → statsmodels
       ├─→ lstm_predictor → PyTorch
       └─→ technical_analyzer → pandas/numpy
       │
       ▼
┌─────────────────┐
│  结果返回LLM     │
└──────┬──────────┘
       │
       │ 多轮迭代...
       ▼
┌─────────────────┐
│  最终分析报告    │
└─────────────────┘
```

## 📝 示例输出

Agent会自动执行完整分析流程:

1. **数据收集**: 获取历史价格和成交量
2. **新闻搜索**: 查找最新市场动态
3. **技术分析**: 计算RSI、MACD等指标
4. **模型预测**: 
   - ARIMA: 统计学习方法
   - LSTM: 深度学习方法
5. **综合建议**: 基于所有信息的投资建议

## ⚠️ 免责声明

本软件仅供教育和研究目的使用，不构成投资建议。股市有风险，投资需谨慎。

## 📄 License

MIT License

## 🤝 Contributing

欢迎提交Issue和Pull Request!
