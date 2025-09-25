import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import openai
import requests
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import datetime
import os
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tensorflow.keras.callbacks import EarlyStopping
import time
import random

# 设置OpenAI API密钥
openai.api_base = "https://api.deepseek.com"
openai.api_key = os.getenv("DEEPSEEK_API_KEY")
if openai.api_key is None:
    print("API key is missing. Please set the environment variable OPENAI_API_KEY.")

# Fetch market data (10 years of data)
def fetch_market_data(symbol, start_date, end_date, max_retries=3, retry_delay=5, proxies=None):
    """
    从Yahoo Finance获取股票数据，包含重试机制、代理支持和错误处理
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        max_retries: 最大重试次数
        retry_delay: 重试延迟时间（秒）
        proxies: 代理设置
    
    Returns:
        DataFrame: 股票数据
    """
    # 设置代理（如果提供）
    if proxies:
        yf.pdr_override()
        import yfinance.shared as yfs
        yfs.session.proxies = proxies
    
    for attempt in range(max_retries):
        try:
            # 添加随机延迟避免频率限制
            if attempt > 0:
                time.sleep(retry_delay + random.uniform(0, 5))
            
            # 设置请求参数
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            print(f"Successfully fetched data for {symbol}")
            return data
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {symbol}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
            else:
                print(f"Failed to fetch data for {symbol} after {max_retries} attempts")
                return None

# Preprocess data, handling missing values and feature engineering
def preprocess_data(data):
    # Use multiple features for prediction
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]  # Use open, high, low, close, and volume as features
    data.interpolate(method='linear', inplace=True)  # Interpolate missing values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

# Train ARIMA model for time series prediction (Using closing price only)
def arima_predict(data_scaled, order=(5, 1, 0)):
    model = ARIMA(data_scaled[:, 3], order=order)  # Use the closing price (index 3) for ARIMA prediction
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=5)  # Forecast the next 5 days
    return forecast

# ARIMA parameter optimization (grid search)
def arima_optimize(data_scaled):
    p_values = range(1, 6)
    d_values = range(1, 3)
    q_values = range(1, 6)
    best_score = float('inf')
    best_params = None
    best_model = None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(data_scaled[:, 3], order=(p, d, q))  # Use closing price
                    model_fit = model.fit()
                    if model_fit.aic < best_score:
                        best_score = model_fit.aic
                        best_params = (p, d, q)
                        best_model = model_fit
                except Exception as e:
                    continue
    print(f"Best ARIMA model parameters: {best_params}")
    return best_model

# Train LSTM model for time series prediction
def lstm_predict(data_scaled, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i-time_step:i, :])  # Use all features for LSTM
        y.append(data_scaled[i, 3])  # Predict the closing price (index 3)
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))  # Reshape for LSTM input

    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=25, batch_size=32, callbacks=[early_stop])

    forecast_input = data_scaled[-time_step:].reshape(1, -1, data_scaled.shape[1])
    forecast = model.predict(forecast_input)
    return forecast

# Call OpenAI API DeepSeek for deep analysis
def analyze_with_deepseek(data):
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",  # Use the correct model name compatible with DeepSeek
            messages=[
                {"role": "system", "content": "你是一个专业的股票市场分析师"},
                {"role": "user", "content": f"Given the following stock data: {data}, 请详细分析并预测股市趋势，提供具体的预测指标和精确的预测结果，请回复数据概览、趋势分析、成交量分析、技术指标分析、预测结果、操作建议、风险提示"}
            ],
            temperature=0.1,
        )
        analysis = response['choices'][0]['message']['content'].strip()
        return analysis
    except Exception as e:
        print(f"Error during DeepSeek API request: {e}")
        return None

# Displaying predictions and analysis in a tkinter window
def show_predictions_in_window(arima_forecast, lstm_forecast, analysis):
    root = tk.Tk()
    root.title("Stock Market Predictions and Analysis")
    root.geometry("800x600")

    # Create a notebook (tabbed interface)
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)

    # ARIMA Results Tab
    arima_frame = ttk.Frame(notebook)
    notebook.add(arima_frame, text="ARIMA 预测结果")
    
    arima_text = tk.Text(arima_frame, wrap=tk.WORD)
    arima_text.pack(fill='both', expand=True)
    arima_text.insert(tk.END, f"ARIMA 预测未来5天股价:\n")
    for i, price in enumerate(arima_forecast, 1):
        arima_text.insert(tk.END, f"第{i}天: {price:.2f}\n")
    arima_text.config(state=tk.DISABLED)

    # LSTM Results Tab
    lstm_frame = ttk.Frame(notebook)
    notebook.add(lstm_frame, text="LSTM 预测结果")
    
    lstm_text = tk.Text(lstm_frame, wrap=tk.WORD)
    lstm_text.pack(fill='both', expand=True)
    lstm_text.insert(tk.END, f"LSTM 预测股价: {lstm_forecast[0]:.2f}\n")
    lstm_text.config(state=tk.DISABLED)

    # Analysis Results Tab
    analysis_frame = ttk.Frame(notebook)
    notebook.add(analysis_frame, text="AI 深度分析")
    
    analysis_text = tk.Text(analysis_frame, wrap=tk.WORD)
    analysis_text.pack(fill='both', expand=True)
    if analysis:
        analysis_text.insert(tk.END, analysis)
    else:
        analysis_text.insert(tk.END, "暂无AI分析结果")
    analysis_text.config(state=tk.DISABLED)

    # Add scrollbars
    for text_widget in [arima_text, lstm_text, analysis_text]:
        scrollbar = tk.Scrollbar(text_widget, orient="vertical", command=text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        text_widget.config(yscrollcommand=scrollbar.set)

    root.mainloop()

# Plotting ARIMA and LSTM predictions
def plot_predictions(data, arima_forecast, lstm_forecast):
    # ARIMA Forecast Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='历史数据', color='blue')
    plt.plot(np.arange(len(data), len(data) + len(arima_forecast)), arima_forecast, label='ARIMA 预测', color='green', marker='o')
    plt.title('股票市场 ARIMA 预测')
    plt.xlabel('时间')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    plt.show()

    # LSTM Forecast Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='历史数据', color='blue')
    plt.plot(np.arange(len(data), len(data) + 1), lstm_forecast, label='LSTM 预测', color='red', marker='o', markersize=10)
    plt.title('股票市场 LSTM 预测')
    plt.xlabel('时间')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    plt.show()

# Create GUI for user input and interaction
def create_gui():
    root = tk.Tk()
    root.title("股票市场预测工具")
    root.geometry("500x300")

    # Add padding to the main window
    main_frame = ttk.Frame(root, padding="20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Configure grid weights for responsive design
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)

    # Input for stock symbol
    ttk.Label(main_frame, text="股票代码 (例如: AAPL, 000001.ss):").grid(row=0, column=0, sticky=tk.W, pady=5)
    symbol_entry = ttk.Entry(main_frame, width=30)
    symbol_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)

    # Input for start date
    ttk.Label(main_frame, text="开始日期 (YYYY-MM-DD):").grid(row=1, column=0, sticky=tk.W, pady=5)
    start_date_entry = ttk.Entry(main_frame, width=30)
    start_date_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
    start_date_entry.insert(0, "2020-01-01")  # Default value

    # Input for end date
    ttk.Label(main_frame, text="结束日期 (YYYY-MM-DD):").grid(row=2, column=0, sticky=tk.W, pady=5)
    end_date_entry = ttk.Entry(main_frame, width=30)
    end_date_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
    end_date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))  # Default to today

    # Progress Bar
    progress = tk.DoubleVar()
    progress_bar = ttk.Progressbar(main_frame, variable=progress, maximum=100, length=300)
    progress_bar.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

    # Status Label
    status_label = ttk.Label(main_frame, text="准备就绪")
    status_label.grid(row=4, column=0, columnspan=2, pady=5)

    def update_status(text):
        status_label.config(text=text)
        root.update_idletasks()

    def on_start_button_click():
        symbol = symbol_entry.get().strip()
        start_date = start_date_entry.get().strip()
        end_date = end_date_entry.get().strip()

        # Validate inputs
        if not symbol or not start_date or not end_date:
            messagebox.showerror("输入错误", "所有字段都是必填项！")
            return

        # Validate date format
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("日期格式错误", "请输入正确的日期格式 (YYYY-MM-DD)")
            return

        # Update status
        update_status("正在获取股票数据...")

        # Fetch data
        data = fetch_market_data(symbol, start_date, end_date)
        if data is None:
            messagebox.showerror("数据获取失败", 
                               "无法获取股票数据。可能的原因：\n"
                               "1. 网络连接问题\n"
                               "2. 股票代码不正确\n"
                               "3. 请求频率过高，请稍后再试\n"
                               "4. Yahoo Finance 服务暂时不可用")
            update_status("数据获取失败")
            return

        update_status("正在处理数据和训练模型...")

        # Data Preprocessing and Model Training
        try:
            data_scaled, scaler = preprocess_data(data)
            arima_forecast = arima_predict(data_scaled)
            lstm_forecast = lstm_predict(data_scaled)
        except Exception as e:
            messagebox.showerror("模型训练错误", f"在训练模型时发生错误: {str(e)}")
            update_status("模型训练失败")
            return

        update_status("正在获取AI深度分析...")

        # Perform Analysis using OpenAI
        analysis = analyze_with_deepseek(data)
        print(f"OpenAI analysis result: {analysis}")

        update_status("完成！正在显示结果...")

        # Show predictions and analysis in window
        show_predictions_in_window(arima_forecast, lstm_forecast, analysis)

        # Plot the predictions
        plot_predictions(data, arima_forecast, lstm_forecast)

        update_status("准备就绪")

    # Start button to trigger prediction
    start_button = ttk.Button(main_frame, text="开始预测", command=on_start_button_click)
    start_button.grid(row=5, column=0, columnspan=2, pady=20)

    # Add tooltips
    symbol_entry.tooltip = "输入股票代码，例如 AAPL (苹果) 或 000001.ss (平安银行)"
    start_date_entry.tooltip = "数据开始日期，格式 YYYY-MM-DD"
    end_date_entry.tooltip = "数据结束日期，格式 YYYY-MM-DD"

    root.mainloop()