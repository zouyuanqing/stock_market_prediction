"""
工具模块 - 提供 LLM Agent 可调用的各种工具
来源声明：部分设计灵感来自 OpenCLAW 和 Hermes Agent 项目
"""

import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from datetime import timedelta
from duckduckgo_search import DDGS


class StockDataTool:
    """获取股票数据工具"""
    
    def __init__(self):
        self.name = "stock_data_fetcher"
        self.description = "获取股票历史数据，包括开盘价、收盘价、最高价、最低价、成交量等"
    
    def execute(self, ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                return {"error": f"No data found for {ticker}"}
            
            return {
                "success": True,
                "ticker": ticker,
                "data": data.to_dict(),
                "summary": {
                    "total_days": len(data),
                    "avg_close": float(data['Close'].mean()),
                    "volatility": float(data['Close'].std())
                }
            }
        except Exception as e:
            return {"error": str(e)}


class WebSearchTool:
    """网络搜索工具"""
    
    def __init__(self):
        self.name = "web_search"
        self.description = "搜索网络信息，获取新闻、市场动态、公司公告等"
    
    def execute(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            return {
                "success": True,
                "query": query,
                "results": [{"title": r.get('title', ''), "url": r.get('href', ''), "snippet": r.get('body', '')} for r in results],
                "count": len(results)
            }
        except Exception as e:
            return {"error": str(e)}


class ARIMATool:
    """ARIMA 模型预测工具 - 参数由 LLM 决定"""
    
    def __init__(self):
        self.name = "arima_predictor"
        self.description = "使用 ARIMA 模型进行时间序列预测，需要指定 (p,d,q) 参数"
    
    def execute(self, ticker: str, start_date: str, end_date: str, p: int = 1, d: int = 1, q: int = 1, forecast_days: int = 5) -> Dict[str, Any]:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if len(data) < 30:
                return {"error": "Insufficient data for ARIMA modeling"}
            
            close_prices = data['Close'].values
            model = ARIMA(close_prices, order=(p, d, q))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=forecast_days)
            
            return {
                "success": True,
                "model_params": {"p": p, "d": d, "q": q},
                "aic": float(fitted_model.aic),
                "forecast": {
                    "dates": [(data.index[-1] + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(forecast_days)],
                    "predicted_prices": forecast.tolist()
                }
            }
        except Exception as e:
            return {"error": str(e)}


class LSTMNetwork(nn.Module):
    """LSTM 神经网络模型"""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class LSTMTool:
    """LSTM 深度学习预测工具 - 架构由 LLM 决定"""
    
    def __init__(self):
        self.name = "lstm_predictor"
        self.description = "使用 LSTM 神经网络进行股价预测，可配置网络架构"
    
    def execute(self, ticker: str, start_date: str, end_date: str, lookback_days: int = 60, hidden_size: int = 50, num_layers: int = 2, epochs: int = 50, forecast_days: int = 5) -> Dict[str, Any]:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if len(data) < lookback_days + 10:
                return {"error": "Insufficient data for LSTM training"}
            
            close_prices = data['Close'].values.reshape(-1, 1)
            mean_price, std_price = np.mean(close_prices), np.std(close_prices)
            scaled_prices = (close_prices - mean_price) / std_price
            
            X, y = [], []
            for i in range(len(scaled_prices) - lookback_days):
                X.append(scaled_prices[i:i+lookback_days].flatten())
                y.append(float(scaled_prices[i+lookback_days][0]))
            
            X = np.array(X).reshape(-1, lookback_days, 1)
            y = np.array(y)
            X_tensor, y_tensor = torch.FloatTensor(X), torch.FloatTensor(y)
            
            model = LSTMNetwork(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
            criterion, optimizer = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=0.001)
            
            losses = []
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_tensor).squeeze()
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            model.eval()
            predictions = []
            current_input = torch.FloatTensor(scaled_prices[-lookback_days:]).unsqueeze(0)
            
            with torch.no_grad():
                for _ in range(forecast_days):
                    pred = model(current_input).squeeze()
                    predictions.append(pred.item())
                    new_input = torch.cat([current_input[:, 1:, :], torch.FloatTensor([[pred.item()]]).unsqueeze(0)], dim=1)
                    current_input = new_input
            
            predicted_prices = [p * std_price + mean_price for p in predictions]
            forecast_dates = [(data.index[-1] + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(forecast_days)]
            
            return {
                "success": True,
                "model_architecture": {"hidden_size": hidden_size, "num_layers": num_layers, "lookback_days": lookback_days},
                "training_info": {"epochs": epochs, "final_loss": float(losses[-1]), "initial_loss": float(losses[0])},
                "forecast": {"dates": forecast_dates, "predicted_prices": predicted_prices}
            }
        except Exception as e:
            return {"error": str(e)}


class TechnicalAnalysisTool:
    """技术分析指标计算工具"""
    
    def __init__(self):
        self.name = "technical_analyzer"
        self.description = "计算各种技术指标：MA, EMA, MACD, RSI, 布林带等"
    
    def execute(self, ticker: str, start_date: str, end_date: str, indicators: List[str] = None) -> Dict[str, Any]:
        if indicators is None:
            indicators = ['MA', 'EMA', 'RSI', 'MACD', 'Bollinger']
        
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                return {"error": f"No data found for {ticker}"}
            
            close = data['Close'].values
            results = {"success": True, "ticker": ticker, "indicators": {}}
            
            if 'RSI' in indicators:
                rsi = self._calculate_rsi(close, 14)
                results["indicators"]["RSI"] = {"current": float(rsi[-1]) if len(rsi) > 0 and rsi[-1] is not None else None, "signal": "overbought" if rsi[-1] and rsi[-1] > 70 else "oversold" if rsi[-1] and rsi[-1] < 30 else "neutral"}
            
            if 'MACD' in indicators:
                macd_line, signal_line, histogram = self._calculate_macd(close)
                results["indicators"]["MACD"] = {"current_signal": "bullish" if histogram[-1] > 0 else "bearish"}
            
            return results
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_rsi(self, prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=period).mean().values
        avg_loss = pd.Series(loss).rolling(window=period).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate([[None], rsi])
    
    def _calculate_macd(self, prices):
        ema12 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line.values, signal_line.values, histogram.values


AVAILABLE_TOOLS = {
    "stock_data_fetcher": StockDataTool,
    "web_search": WebSearchTool,
    "arima_predictor": ARIMATool,
    "lstm_predictor": LSTMTool,
    "technical_analyzer": TechnicalAnalysisTool
}


def get_tool(tool_name: str):
    if tool_name in AVAILABLE_TOOLS:
        return AVAILABLE_TOOLS[tool_name]()
    raise ValueError(f"Unknown tool: {tool_name}")


def list_tools() -> List[Dict[str, str]]:
    return [{"name": tool.name, "description": tool.description} for tool in [cls() for cls in AVAILABLE_TOOLS.values()]]
