# Stock Market Prediction Tool

## 项目简介

这是一个基于 ARIMA 和 LSTM 模型的股票市场预测工具。该工具能够从 Yahoo Finance 获取股票数据，利用 ARIMA 和 LSTM 模型进行预测，并通过 OpenAI 的 DeepSeek API 提供深度分析。工具还提供了一个简单的 GUI 界面，方便用户进行交互。

## 功能特性

- **数据获取**：从 Yahoo Finance 获取股票数据，并支持多种股票数据源。
- **数据预处理**：包括缺失值处理、特征工程、数据规范化等预处理操作，确保模型训练的高效性。
- **ARIMA 模型**：基于 ARIMA 模型进行时间序列分析，提供股票未来走势的预测。
- **LSTM 模型**：采用 LSTM（长短期记忆）神经网络模型，利用深度学习技术对股票价格进行预测。
- **OpenAI 深度分析**：调用 OpenAI 的 DeepSeek API 进行对预测结果和股票市场的深度分析。
- **GUI 界面**：提供简洁直观的图形用户界面，用户可以方便地输入参数并查看预测结果。

## 安装步骤

1. **克隆项目仓库**：
   ```bash
   git clone https://github.com/zouyuanqing/stock_market_prediction.git
   cd stock_market_prediction
   ```

2. **安装依赖**：
   在项目根目录下，使用以下命令安装所需的 Python 库：
   ```bash
   pip install -r requirements.txt
   ```

3. **运行项目**：
   启动项目的主程序，打开 GUI 界面：
   ```bash
   python -m stock_market_prediction.main
   ```

## 使用方法

1. 运行项目后，会弹出一个 GUI 窗口。
2. 在界面中输入股票代码（如 AAPL）、开始日期和结束日期。
3. 点击 "Start" 按钮，程序将自动从 Yahoo Finance 获取数据，并开始训练 ARIMA 和 LSTM 模型。
4. 预测结果将显示在界面中，包括 ARIMA 和 LSTM 的预测值。
5. OpenAI 的 DeepSeek API 会对预测结果进行深度分析，生成相应的分析报告，展示在界面上。

## 项目结构

```
stock_market_prediction/
├── stock_market_prediction/   # 主包目录
│   ├── __init__.py            # 包初始化文件
│   ├── main.py                # 主程序入口
│   ├── utils.py               # 业务逻辑模块
├── tests/                     # 测试目录
│   ├── __init__.py            # 测试包初始化文件
│   ├── test_utils.py          # 测试模块
├── requirements.txt           # 依赖文件
├── setup.py                   # 打包配置文件
├── README.md                  # 项目说明文档
```

## 依赖说明

- **Python 3.8+**
- **pandas**：数据处理与分析
- **numpy**：科学计算
- **matplotlib**：绘图工具
- **yfinance**：获取 Yahoo Finance 数据
- **openai**：调用 OpenAI API
- **requests**：处理 HTTP 请求
- **statsmodels**：ARIMA 模型实现
- **scikit-learn**：机器学习工具包
- **tensorflow**：深度学习框架

## 贡献指南

欢迎贡献代码或报告问题！请按照以下步骤操作：

1. **Fork 项目仓库**。
2. **创建一个新的分支**：
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **提交更改**：
   ```bash
   git commit -m "Add your feature"
   ```
4. **推送分支**：
   ```bash
   git push origin feature/your-feature-name
   ```
5. **提交 Pull Request**。

## 许可证

本项目基于 MIT 许可证开源。详情请参阅 LICENSE 文件。

## 联系方式

作者：[zouyuanqing](mailto:zou.yuanqing@foxmail.com)  邹源清
邮箱：[zou.yuanqing@foxmail.com](mailto:zou.yuanqing@foxmail.com)
