# 📈 Stock Predictor - 股票市场预测工具

基于 **Tauri 2.0 + React 18 + TypeScript** 构建的现代化股票市场预测工具，集成 AI 智能分析。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue.svg)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![Tauri](https://img.shields.io/badge/Tauri-2.0-FFC131.svg)](https://tauri.app/)
[![Rust](https://img.shields.io/badge/Rust-1.70+-DEA584.svg)](https://www.rust-lang.org/)

---

## ✨ 功能特性

### 📊 数据分析
- **多数据源支持**: Yahoo Finance、Stooq，一键切换
- **技术指标**: MA5/MA10/MA20、RSI、MACD
- **交易信号**: 自动生成买卖信号提示

### 🤖 AI 智能分析
- **多模型支持**: DeepSeek、OpenAI 及兼容 API
- **自动模型发现**: 填入 API Key 后自动拉取可用模型列表
- **Markdown 渲染**: 结构化分析报告，支持表格、代码块等

### 📈 可视化
- **交互式图表**: ECharts 驱动，支持缩放、拖拽
- **双图联动**: 价格走势 + 成交量同步展示
- **深色模式**: 完美适配浅色/深色主题

### 🎨 现代化 UI
- **响应式设计**: 适配各种屏幕尺寸
- **流畅动画**: 平滑的过渡和交互效果
- **组件化架构**: React 组件，易于扩展

---

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| **前端框架** | React 18 + TypeScript |
| **构建工具** | Vite 5 |
| **UI 框架** | Tailwind CSS 3 |
| **图表库** | ECharts 5 |
| **Markdown** | react-markdown + remark-gfm |
| **桌面框架** | Tauri 2.0 |
| **后端语言** | Rust |
| **HTTP 客户端** | reqwest |
| **包管理器** | pnpm |

---

## 📦 安装

### 前置要求

- [Node.js](https://nodejs.org/) 18+
- [Rust](https://www.rust-lang.org/tools/install) 1.70+
- [pnpm](https://pnpm.io/) (推荐)

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/zouyuanqing/stock-predictor.git
cd stock-predictor

# 2. 安装依赖
pnpm install

# 3. 开发模式运行
pnpm tauri dev

# 4. 构建生产版本
pnpm tauri build
```

### Windows 用户注意

如果 `rustc` 命令找不到，需要添加到 PATH：

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
```

---

## 🚀 快速开始

### 1. 配置 AI 分析（可选）

点击界面顶部的 **"展开配置"** 按钮：

- **API Base URL**: 填入 API 地址
  - DeepSeek: `https://api.deepseek.com`
  - OpenAI: `https://api.openai.com/v1`
  
- **API Key**: 填入你的 API 密钥
  - 填入后自动拉取可用模型列表
  - 不填则跳过 AI 分析，仅显示技术指标

### 2. 查询股票

1. 输入股票代码（如 `AAPL`、`000001.ss`）
2. 选择开始/结束日期
3. 选择数据源
4. 点击 **"🚀 开始预测"**

### 3. 查看结果

- **左侧**: 价格走势图表 + 成交量
- **右侧**: 技术指标分析 + 交易信号
- **下方**: AI 深度分析报告（Markdown 格式）

---

## 📁 项目结构

```
stock-predictor/
├── src/                          # 前端源码 (React + TypeScript)
│   ├── components/               # React 组件
│   │   ├── Header.tsx            # 导航头部
│   │   ├── StockChart.tsx        # ECharts 图表
│   │   ├── PredictionPanel.tsx   # 技术指标面板
│   │   └── AnalysisPanel.tsx     # AI 分析面板 (Markdown)
│   ├── types/                    # TypeScript 类型定义
│   ├── App.tsx                   # 主应用组件
│   ├── main.tsx                  # 入口文件
│   └── index.css                 # 全局样式
├── src-tauri/                    # Tauri 后端 (Rust)
│   ├── src/
│   │   ├── lib.rs                # 核心业务逻辑
│   │   └── main.rs               # 入口文件
│   ├── icons/                    # 应用图标
│   ├── Cargo.toml                # Rust 依赖配置
│   └── tauri.conf.json           # Tauri 配置
├── package.json                  # Node.js 依赖
├── tailwind.config.js            # Tailwind CSS 配置
├── vite.config.ts                # Vite 配置
├── CHANGELOG.md                  # 更新日志
└── README.md                     # 项目文档
```

---

## 🆚 版本对比

### v2.0 (TypeScript + Tauri) vs v1.0 (Python + Tkinter)

| 特性 | v1.0 (Python) | v2.0 (TypeScript) |
|------|--------------|-------------------|
| **UI 美观度** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **启动速度** | ~3s | ~0.5s |
| **打包体积** | ~150MB | ~15MB |
| **内存占用** | ~200MB | ~80MB |
| **图表交互** | 静态 | 动态交互 |
| **Markdown** | ❌ | ✅ |
| **模型选择** | ❌ | ✅ |
| **深色模式** | ❌ | ✅ |
| **跨平台** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🔧 开发指南

### 添加新组件

```tsx
// src/components/MyComponent.tsx
export function MyComponent() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
      {/* 组件内容 */}
    </div>
  );
}
```

### 添加 Tauri 命令

```rust
// src-tauri/src/lib.rs
#[tauri::command]
async fn my_command() -> Result<String, String> {
    Ok("Hello from Rust!".to_string())
}
```

### 调用 Tauri 命令

```tsx
import { invoke } from "@tauri-apps/api/core";

const result = await invoke<string>("my_command");
```

---

## 📊 API 兼容性

本项目支持所有兼容 OpenAI API 格式的服务：

| 服务 | API Base URL |
|------|--------------|
| DeepSeek | `https://api.deepseek.com` |
| OpenAI | `https://api.openai.com/v1` |
| 月之暗面 | `https://api.moonshot.cn/v1` |
| 智谱 AI | `https://open.bigmodel.cn/api/paas/v4` |
| 零一万物 | `https://api.lingyiwanwu.com/v1` |

---

## 🐛 常见问题

### Q: 数据获取失败？
A: 
- 检查网络连接
- 确认股票代码正确
- 尝试切换数据源（Yahoo/Stooq）

### Q: AI 分析不可用？
A:
- 确认已配置 API Key
- 检查 API Base URL 是否正确
- 点击"刷新模型"重新加载模型列表

### Q: Rust 编译失败？
A:
- 确保 Rust 已安装: `rustc --version`
- 添加到 PATH: `$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"`

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

## 👨‍💻 作者

**邹源清** - [zou.yuanqing@foxmail.com](mailto:zou.yuanqing@foxmail.com)

- GitHub: [@zouyuanqing](https://github.com/zouyuanqing)

---

## 🙏 致谢

- [Tauri](https://tauri.app/) - 跨平台桌面应用框架
- [React](https://react.dev/) - 用户界面库
- [ECharts](https://echarts.apache.org/) - 数据可视化库
- [Tailwind CSS](https://tailwindcss.com/) - 实用优先的 CSS 框架
- [DeepSeek](https://deepseek.com/) - AI 模型服务

---

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=zouyuanqing/stock-predictor&type=Date)](https://star-history.com/#zouyuanqing/stock-predictor&Date)
