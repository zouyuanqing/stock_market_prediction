# 股票市场预测工具 (Stock Market Prediction Tool)

## 项目简介

这是一个基于 ARIMA 和 LSTM 模型的股票市场预测工具。该工具能够从多个数据源获取股票数据，利用 ARIMA 和 LSTM 模型进行预测，并通过 DeepSeek API 提供深度分析。工具还提供了一个增强的 GUI 界面，方便用户进行交互和配置。

本项目经过重大更新，解决了 Yahoo Finance 数据获取频率限制问题，增强了稳定性和用户体验。需要注意的是，Yahoo Finance 和 Stooq 在某些地区可能存在访问限制，如果遇到此类问题，建议使用代理或配置其他数据源。

## 功能特性

- **多数据源支持**：支持 Yahoo Finance 和 Stooq 等多个数据源，避免单一数据源不可用的问题
- **智能重试机制**：针对频率限制错误自动增加等待时间，提高数据获取成功率
- **数据预处理**：包括缺失值处理、特征工程、数据规范化等预处理操作，确保模型训练的高效性
- **ARIMA 模型**：基于 ARIMA 模型进行时间序列分析，提供股票未来走势的预测
- **LSTM 模型**：采用 LSTM（长短期记忆）神经网络模型，利用深度学习技术对股票价格进行预测
- **AI 深度分析**：调用 DeepSeek API 进行对预测结果和股票市场的深度分析
- **Markdown 渲染**：AI 分析结果以格式化方式显示，支持标题、列表等元素
- **可配置 API**：支持通过 .env 文件或 GUI 界面配置 API 参数
- **增强 GUI 界面**：提供直观的图形用户界面，支持 API 配置和数据源选择

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

3. **配置 API 密钥**（可选）：
   复制 `.env.example` 文件并重命名为 `.env`，然后在其中设置您的 DeepSeek API 密钥：
   ```bash
   cp .env.example .env
   ```
   编辑 `.env` 文件，设置您的 API 密钥：
   ```env
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   DEEPSEEK_API_BASE=https://api.deepseek.com
   ```

4. **运行项目**：
   启动项目的主程序，打开 GUI 界面：
   ```bash
   python run.py
   ```

## 使用方法

1. 运行项目后，会弹出一个 GUI 窗口。
2. （可选）在 API 设置区域配置您的 API Base URL 和 API Key
3. 在界面中输入股票代码（如 AAPL、000001.ss）、开始日期和结束日期
4. 选择数据源（Yahoo Finance 或 Stooq）
5. 点击 "开始预测" 按钮，程序将自动获取数据并开始训练 ARIMA 和 LSTM 模型
6. 预测结果将显示在界面中，包括 ARIMA 和 LSTM 的预测值
7. OpenAI 的 DeepSeek API 会对预测结果进行深度分析，生成相应的分析报告，以格式化方式展示在界面上

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
├── setup.py                  # 打包配置文件
├── run.py                    # 运行脚本
├── .env.example              # API 配置示例文件
├── README.md                 # 项目说明文档
├── UPDATE_LOG.md             # 更新日志
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
- **python-dotenv**：环境变量加载
- **markdown**：Markdown 渲染支持

## 配置说明

### API 配置

您可以通过两种方式配置 API：

1. **环境文件配置**：
   创建 `.env` 文件并设置以下变量：
   ```env
   # DeepSeek API 配置
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   DEEPSEEK_API_BASE=https://api.deepseek.com
   
   # OpenAI API 配置（可选）
   # OPENAI_API_KEY=your_openai_api_key_here
   # OPENAI_API_BASE=https://api.openai.com/v1
   ```

2. **GUI 界面配置**：
   在 GUI 的 API 设置区域直接输入 API Base URL 和 API Key，然后点击"保存API设置"按钮。

## 故障排除

### 常见问题

1. **Yahoo Finance 数据获取失败**：
   - 项目现在支持多数据源，如果 Yahoo Finance 不可用，会自动尝试 Stooq
   - 增加了智能重试机制，会自动处理频率限制问题

2. **AI 分析功能不可用**：
   - 确保已在 `.env` 文件中正确配置 API 密钥
   - 或在 GUI 中配置并保存 API 设置

3. **LSTM 预测错误**：
   - 确保输入的数据时间范围足够长（至少60天）
   - 检查股票代码是否正确

## 更新日志

详细更新信息请查看 [UPDATE_LOG.md](UPDATE_LOG.md) 文件。

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

作者：[zouyuanqing](mailto:zou.yuanqing@foxmail.com) 邹源清
邮箱：[zou.yuanqing@foxmail.com](mailto:zou.yuanqing@foxmail.com)