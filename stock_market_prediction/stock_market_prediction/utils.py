"""
Stock Market Prediction Module - Refactored Version
Improved quantitative analysis logic with better reliability
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import time
import random
from dotenv import load_dotenv
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 可选的 TensorFlow 导入
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("警告：TensorFlow 未安装，LSTM 预测功能将不可用")


class StockPredictor:
    """
    股票市场预测器类 - 重构版本
    提供可靠的数据获取、预处理、模型训练和预测功能
    """
    
    def __init__(self, api_key: Optional[str] = None, api_base: str = "https://api.deepseek.com"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        self.scaler: Optional[MinMaxScaler] = None
        self.api_configured = False
        
        if self.api_key:
            self._setup_api()
    
    def _setup_api(self):
        """配置 DeepSeek API"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            self.api_configured = True
            print(f"DeepSeek API 已配置：{self.api_base}")
        except Exception as e:
            print(f"API 配置失败：{e}")
            self.api_configured = False
    
    def fetch_market_data(
        self, symbol: str, start_date: str, end_date: str,
        max_retries: int = 3, retry_delay: int = 5
    ) -> Optional[pd.DataFrame]:
        """获取股票市场数据"""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(retry_delay * (2 ** (attempt - 1)))
                
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if data is not None and not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required_cols:
                        if col not in data.columns:
                            data[col] = data.get('Close', 0) if col != 'Volume' else 0
                    
                    return data[required_cols]
            except Exception as e:
                print(f"尝试 {attempt + 1} 失败：{e}")
        
        return None
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
        """预处理数据"""
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        data_processed = data[features].copy()
        
        data_processed.interpolate(method='linear', inplace=True)
        data_processed.fillna(method='bfill', inplace=True)
        data_processed.fillna(method='ffill', inplace=True)
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = self.scaler.fit_transform(data_processed)
        
        return data_scaled, self.scaler
    
    def arima_predict(self, data_scaled: np.ndarray, order: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """ARIMA 模型预测"""
        close_prices = data_scaled[:, 3]
        
        if order is None:
            order = self._optimize_arima(close_prices)
        
        try:
            model = ARIMA(close_prices, order=order)
            model_fit = model.fit()
            forecast_scaled = model_fit.forecast(steps=5)
            
            dummy = np.zeros((5, 5))
            dummy[:, 3] = forecast_scaled
            forecast = self.scaler.inverse_transform(dummy)[:, 3]
            return forecast
        except Exception as e:
            print(f"ARIMA 预测失败：{e}")
            last_close = data_scaled[-5:, 3].mean()
            dummy = np.zeros((5, 5))
            dummy[:, 3] = last_close
            return self.scaler.inverse_transform(dummy)[:, 3]
    
    def _optimize_arima(self, close_prices: np.ndarray) -> Tuple[int, int, int]:
        """优化 ARIMA 参数"""
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)
        
        best_score = float('inf')
        best_params = (1, 1, 1)
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        model = ARIMA(close_prices, order=(p, d, q))
                        model_fit = model.fit()
                        if model_fit.aic < best_score:
                            best_score = model_fit.aic
                            best_params = (p, d, q)
                    except:
                        continue
        
        print(f"最佳 ARIMA 参数：{best_params}")
        return best_params
    
    def lstm_predict(self, data_scaled: np.ndarray, time_step: int = 20) -> np.ndarray:
        """LSTM 模型预测"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow 不可用，使用简单移动平均作为替代")
            last_close = data_scaled[-5:, 3].mean()
            dummy = np.zeros((1, 5))
            dummy[:, 3] = last_close
            return self.scaler.inverse_transform(dummy)[:, 3]
        
        if len(data_scaled) < time_step:
            time_step = max(10, len(data_scaled) // 2)
        
        X, y = [], []
        for i in range(time_step, len(data_scaled)):
            X.append(data_scaled[i - time_step:i, :])
            y.append(data_scaled[i, 3])
        
        X, y = np.array(X), np.array(y)
        
        if len(X) == 0:
            dummy = np.zeros((1, 5))
            dummy[:, 3] = 0.5
            return self.scaler.inverse_transform(dummy)[:, 3]
        
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(units=30, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=30, batch_size=min(16, len(X)), callbacks=[early_stop], verbose=0)
        
        forecast_input = data_scaled[-time_step:].reshape(1, time_step, data_scaled.shape[1])
        forecast_scaled = model.predict(forecast_input, verbose=0)
        
        dummy = np.zeros((1, 5))
        dummy[:, 3] = forecast_scaled.flatten()
        return self.scaler.inverse_transform(dummy)[:, 3]
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算技术指标"""
        indicators = {}
        
        indicators['MA5'] = float(data['Close'].rolling(window=5).mean().iloc[-1])
        indicators['MA10'] = float(data['Close'].rolling(window=10).mean().iloc[-1])
        indicators['MA20'] = float(data['Close'].rolling(window=20).mean().iloc[-1])
        indicators['EMA12'] = float(data['Close'].ewm(span=12).mean().iloc[-1])
        indicators['EMA26'] = float(data['Close'].ewm(span=26).mean().iloc[-1])
        indicators['MACD'] = float(indicators['EMA12'] - indicators['EMA26'])
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
        rs = gain / loss if loss != 0 else 0
        indicators['RSI'] = float(100 - (100 / (1 + rs)))
        
        ma20 = data['Close'].rolling(window=20).mean().iloc[-1]
        std20 = data['Close'].rolling(window=20).std().iloc[-1]
        indicators['Bollinger_Upper'] = float(ma20 + 2 * std20)
        indicators['Bollinger_Lower'] = float(ma20 - 2 * std20)
        
        indicators['Current_Price'] = float(data['Close'].iloc[-1])
        indicators['Avg_Volume'] = float(data['Volume'].rolling(window=20).mean().iloc[-1])
        indicators['Current_Volume'] = float(data['Volume'].iloc[-1])
        indicators['Volume_Ratio'] = float(indicators['Current_Volume'] / indicators['Avg_Volume']) if indicators['Avg_Volume'] != 0 else 1.0
        
        return indicators
    
    def analyze_with_deepseek(self, data: pd.DataFrame, arima_forecast: np.ndarray, 
                              lstm_forecast: np.ndarray, technical_indicators: Dict[str, Any]) -> Optional[str]:
        """使用 DeepSeek API 进行深度分析"""
        if not self.api_configured or not self.api_key:
            return None
        
        try:
            prompt = f"""作为专业股票分析师，请基于以下数据分析：

当前价格：{technical_indicators['Current_Price']:.2f}
近期趋势：{'上涨' if data['Close'].iloc[-1] > data['Close'].iloc[-5] else '下跌'}

模型预测:
- ARIMA 未来 5 天：{[f'{x:.2f}' for x in arima_forecast]}
- LSTM 预测：{lstm_forecast[0]:.2f}

技术指标:
- MA5/10/20: {technical_indicators['MA5']:.2f}/{technical_indicators['MA10']:.2f}/{technical_indicators['MA20']:.2f}
- RSI: {technical_indicators['RSI']:.2f}
- MACD: {technical_indicators['MACD']:.4f}
- 成交量比率：{technical_indicators['Volume_Ratio']:.2f}

请提供：1.数据概览 2.趋势分析 3.技术指标解读 4.预测对比 5.操作建议 6.风险提示"""

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是专业的股票市场分析师。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"DeepSeek API 请求失败：{e}")
            return None
    
    def run_full_analysis(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """执行完整分析流程"""
        results = {'symbol': symbol, 'success': False, 'error': None}
        
        try:
            print(f"正在获取 {symbol} 的数据...")
            data = self.fetch_market_data(symbol, start_date, end_date)
            
            if data is None or data.empty:
                results['error'] = "无法获取股票数据"
                return results
            
            print("正在预处理数据...")
            data_scaled, _ = self.preprocess_data(data)
            
            print("正在计算技术指标...")
            indicators = self.calculate_technical_indicators(data)
            
            print("正在运行 ARIMA 模型...")
            arima_forecast = self.arima_predict(data_scaled)
            
            print("正在运行 LSTM 模型...")
            lstm_forecast = self.lstm_predict(data_scaled)
            
            ai_analysis = None
            if self.api_configured:
                print("正在获取 AI 分析...")
                ai_analysis = self.analyze_with_deepseek(data, arima_forecast, lstm_forecast, indicators)
            
            results.update({
                'success': True,
                'data': data,
                'arima_forecast': arima_forecast,
                'lstm_forecast': lstm_forecast,
                'technical_indicators': indicators,
                'ai_analysis': ai_analysis
            })
            print("分析完成！")
            
        except Exception as e:
            results['error'] = str(e)
            print(f"分析出错：{e}")
        
        return results


# 向后兼容的函数接口
def fetch_market_data(symbol, start_date, end_date, max_retries=3, retry_delay=5):
    predictor = StockPredictor()
    return predictor.fetch_market_data(symbol, start_date, end_date, max_retries, retry_delay)

def preprocess_data(data):
    predictor = StockPredictor()
    return predictor.preprocess_data(data)

def arima_predict(data_scaled, order=None):
    predictor = StockPredictor()
    dummy = np.random.rand(10, 5)
    predictor.scaler = MinMaxScaler(feature_range=(0, 1))
    predictor.scaler.fit(dummy)
    return predictor.arima_predict(data_scaled, order)

def lstm_predict(data_scaled, time_step=20):
    predictor = StockPredictor()
    dummy = np.random.rand(100, 5)
    predictor.scaler = MinMaxScaler(feature_range=(0, 1))
    predictor.scaler.fit(dummy)
    return predictor.lstm_predict(data_scaled, time_step)

def analyze_with_deepseek(data):
    predictor = StockPredictor()
    return predictor.analyze_with_deepseek(data, np.array([0]*5), np.array([0]), {})

def create_gui():
    from stock_market_prediction.gui import create_main_gui
    create_main_gui()

def show_predictions_in_window(arima_forecast, lstm_forecast, analysis):
    from stock_market_prediction.gui import display_results
    display_results(arima_forecast, lstm_forecast, analysis)

def plot_predictions(data, arima_forecast, lstm_forecast):
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(data['Close'].values, label='历史数据', color='blue')
    arima_extended = np.concatenate([data['Close'].values[-5:], arima_forecast])
    plt.plot(range(len(data)-5, len(data)+5), arima_extended, label='ARIMA 预测', color='green', marker='o')
    plt.title('ARIMA 预测结果')
    plt.xlabel('时间')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(data['Close'].values, label='历史数据', color='blue')
    plt.scatter(len(data)-1, lstm_forecast[0], color='red', s=100, label='LSTM 预测', zorder=5)
    plt.title('LSTM 预测结果')
    plt.xlabel('时间')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
