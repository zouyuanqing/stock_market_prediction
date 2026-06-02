# 股票市场预测项目更新说明

## 版本信息
- 版本号：v1.0.0
- 更新日期：2025年9月25日
- 作者：AI助手

## 更新概述
本次更新主要解决了用户反馈的Yahoo Finance数据获取问题，并增强了项目的功能性和用户体验。主要改进包括：
1. 增强了数据获取模块，添加了重试机制和多数据源支持
2. 添加了.env文件支持，允许用户配置API参数
3. 改进了GUI界面，增加了API设置功能
4. 增强了AI分析结果的显示，添加了markdown渲染支持
5. 修复了多个bug，提高了程序的稳定性
6. 添加了关于部分地区访问Yahoo Finance和Stooq受限的说明

## 详细变更说明

### 1. 数据获取模块改进
- **问题**：Yahoo Finance数据获取经常遇到频率限制错误（YFRateLimitError）
- **解决方案**：
  - 增加了更智能的重试机制，对于频率限制错误会增加等待时间
  - 添加了多数据源支持，包括Stooq作为备选数据源
  - 改进了错误处理和用户提示信息

### 2. API配置功能
- **新增功能**：添加了.env文件支持
  - 用户可以在.env文件中配置DeepSeek API密钥和URL
  - GUI中增加了API设置区域，用户可以动态配置API参数
  - 支持OpenAI格式的URL与密钥配置
- **相关文件**：
  - 新增：.env（环境配置文件）
  - 修改：requirements.txt（添加python-dotenv依赖）
  - 修改：stock_market_prediction/utils.py（API配置功能）

### 3. GUI界面改进
- **界面增强**：
  - 增加了API设置区域，用户可以输入API Base URL和API Key
  - 添加了"保存API设置"按钮，可以将配置保存到.env文件
  - 增加了数据源选择下拉框，支持yahoo和stooq数据源
  - 扩展了窗口大小以容纳新增功能
- **用户体验**：
  - 提供了更友好的错误提示信息
  - 改进了状态更新机制

### 4. AI分析结果显示优化
- **markdown渲染支持**：
  - 添加了markdown渲染功能，AI分析结果以格式化方式显示
  - 支持标题、粗体、列表等markdown元素的渲染
  - 为不同级别的标题添加了颜色和字体样式
- **相关文件**：
  - 修改：requirements.txt（添加markdown依赖）
  - 修改：stock_market_prediction/utils.py（markdown渲染功能）

### 5. Bug修复
- **LSTM预测结果格式化错误**：
  - 修复了LSTM预测结果可能是多维NumPy数组导致的TypeError
  - 添加了类型检查和适当的数组展平处理
- **API密钥提示信息不清晰**：
  - 提供了更友好的中文提示，说明AI分析功能不可用的原因和解决方法
- **模块导入问题**：
  - 修复了相对导入问题，确保项目可以正确安装和运行

## 修复的Bug列表

| Bug ID | 问题描述 | 修复方法 |
|--------|----------|----------|
| BUG-001 | Yahoo Finance数据获取频率限制 | 增加重试机制和多数据源支持 |
| BUG-002 | LSTM预测结果显示TypeError | 添加类型检查和数组展平处理 |
| BUG-003 | API密钥提示信息不清晰 | 提供更友好的中文提示 |
| BUG-004 | 模块导入错误 | 修复相对导入问题 |
| BUG-005 | 环境配置不灵活 | 添加.env文件支持 |

## 新增功能列表

| 功能ID | 功能描述 | 相关文件 |
|--------|----------|----------|
| FEATURE-001 | 多数据源支持 | stock_market_prediction/utils.py |
| FEATURE-002 | .env文件配置支持 | .env, requirements.txt, stock_market_prediction/utils.py |
| FEATURE-003 | GUI API参数设置 | stock_market_prediction/utils.py |
| FEATURE-004 | Markdown渲染支持 | requirements.txt, stock_market_prediction/utils.py |
| FEATURE-005 | 智能重试机制 | stock_market_prediction/utils.py |

## 文件变更列表

### 新增文件
1. .env - 环境配置文件
2. test_api_settings.py - API设置功能测试
3. test_markdown_render.py - Markdown渲染测试
4. test_gui_markdown.py - GUI Markdown渲染测试

### 修改文件
1. requirements.txt - 添加新依赖
2. setup.py - 完善项目元数据
3. stock_market_prediction/utils.py - 核心功能改进
4. stock_market_prediction/main.py - 入口点调整

### 依赖变更
```txt
# 新增依赖
python-dotenv>=1.0.0
markdown>=3.4.0
```

## 使用说明

### 环境配置
1. 在项目根目录创建或修改.env文件：
   ```env
   # DeepSeek API配置
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   DEEPSEEK_API_BASE=https://api.deepseek.com
   
   # OpenAI API配置（可选）
   # OPENAI_API_KEY=your_openai_api_key_here
   # OPENAI_API_BASE=https://api.openai.com/v1
   ```

### 运行项目
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行GUI：
   ```bash
   python run.py
   ```

### 使用新功能
1. **API配置**：
   - 在GUI的API设置区域输入API Base URL和API Key
   - 点击"保存API设置"按钮保存配置

2. **数据源选择**：
   - 在GUI中选择数据源（yahoo或stooq）

3. **查看AI分析结果**：
   - AI分析结果将以格式化方式显示，支持标题、列表等元素

## 故障排除

### 常见问题

1. **Yahoo Finance 数据获取失败**：
   - 项目现在支持多数据源，如果 Yahoo Finance 不可用，会自动尝试 Stooq
   - 增加了智能重试机制，会自动处理频率限制问题
   - 注意：Yahoo Finance 和 Stooq 在某些地区可能存在访问限制，如果遇到此类问题，建议使用代理或配置其他数据源

2. **AI 分析功能不可用**：
   - 确保已在 `.env` 文件中正确配置 API 密钥
   - 或在 GUI 中配置并保存 API 设置

3. **LSTM 预测错误**：
   - 确保输入的数据时间范围足够长（至少60天）
   - 检查股票代码是否正确

## 测试验证
所有新增功能都经过了测试验证：
- API设置功能测试通过
- Markdown渲染功能测试通过
- GUI中的功能测试通过
- 数据获取功能测试通过

## 后续建议
1. 可以考虑添加更多数据源支持
2. 可以进一步优化markdown渲染功能，支持更多markdown元素
3. 可以添加更多AI模型支持