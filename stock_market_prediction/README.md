# Stock Market Prediction Tool v2.0

## 项目简介

这是一个基于 ARIMA 和 LSTM 模型的股票市场预测工具，经过全面重构，提供更可靠的量化分析逻辑。该工具能够从 Yahoo Finance 获取股票数据，利用多种模型进行预测，并通过 DeepSeek API 提供深度分析。

## 主要改进 (v2.0)

### 核心重构
- **面向对象设计**: 引入 `StockPredictor` 类，封装完整的预测流程
- **改进的量化分析**: 
  - 自动 ARIMA 参数优化（基于 AIC 准则）
  - 更可靠的数据预处理和反规范化处理
  - 完整的错误处理和降级策略
- **技术指标系统**: 新增完整的技术指标计算模块
  - 移动平均线 (MA5, MA10, MA20)
  - 指数移动平均线 (EMA12, EMA26)
  - MACD、RSI、布林带
  - 成交量分析

### API 集成改进
- 支持新版 OpenAI API 格式
- 更智能的 prompt 工程，包含技术指标和模型预测对比
- 优雅降级：API 不可用时不影响核心功能

### 模块化设计
- 可选的 TensorFlow 依赖（LSTM 功能）
- 向后兼容的函数接口
- 清晰的模块分离（utils、gui）

## 安装步骤

1. **克隆项目仓库**：
   ```bash
   git clone https://github.com/zouyuanqing/stock_market_prediction.git
   cd stock_market_prediction
   ```

2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
   
   可选：安装 TensorFlow 以启用 LSTM 预测
   ```bash
   pip install tensorflow-cpu
   ```

3. **配置 API（可选）**：
   在项目根目录创建 `.env` 文件：
   ```
   DEEPSEEK_API_KEY=your_api_key_here
   DEEPSEEK_API_BASE=https://api.deepseek.com
   ```

4. **运行项目**：
   ```bash
   python -m stock_market_prediction.main
   ```

## 使用方法

### 命令行使用

```python
from stock_market_prediction.utils import StockPredictor

# 创建预测器（可传入 API key）
predictor = StockPredictor(api_key="your_api_key")

# 执行完整分析
results = predictor.run_full_analysis(
    symbol="AAPL",
    start_date="2023-01-01",
    end_date="2024-01-01"
)

# 查看结果
if results['success']:
    print("ARIMA 预测:", results['arima_forecast'])
    print("LSTM 预测:", results['lstm_forecast'])
    print("技术指标:", results['technical_indicators'])
    print("AI 分析:", results['ai_analysis'])
```

### GUI 使用

运行后会出现图形界面，输入参数后点击"开始分析"即可。

## 项目结构

```
stock_market_prediction/
├── stock_market_prediction/
│   ├── __init__.py
│   ├── main.py              # 程序入口
│   ├── utils.py             # 核心预测逻辑（StockPredictor 类）
│   └── gui.py               # GUI 界面
├── tests/
│   ├── test_utils.py        # 原有测试
│   └── test_stock_predictor.py  # 新测试
├── requirements.txt
├── README.md
└── UPDATE_LOG.md
```

## 依赖说明

**必需**:
- Python 3.8+
- pandas
- numpy
- matplotlib
- yfinance
- statsmodels
- scikit-learn
- python-dotenv
- openai

**可选**:
- tensorflow (用于 LSTM 预测)

## 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

MIT License

## 联系方式

作者：邹源清 (zouyuanqing)  
邮箱：zou.yuanqing@foxmail.com
