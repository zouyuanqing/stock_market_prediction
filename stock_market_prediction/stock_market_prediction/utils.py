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

    # Creating the message content
    arima_message = f"ARIMA forecast for the next 5 days: {arima_forecast}"
    lstm_message = f"LSTM forecast for the stock market: {lstm_forecast}"
    analysis_message = f"OpenAI Analysis Result:\n{analysis}"

    # Display in a messagebox
    messagebox.showinfo("Forecast Results", f"{arima_message}\n\n{lstm_message}\n\n{analysis_message}")
    root.mainloop()

# Plotting ARIMA and LSTM predictions
def plot_predictions(data, arima_forecast, lstm_forecast):
    # ARIMA Forecast Plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Historical Data', color='blue')
    plt.plot(np.arange(len(data), len(data) + len(arima_forecast)), arima_forecast, label='ARIMA Forecast', color='green')
    plt.title('Stock Market ARIMA Forecast')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # LSTM Forecast Plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Historical Data', color='blue')
    plt.plot(np.arange(len(data), len(data) + 1), lstm_forecast, label='LSTM Forecast', color='red')
    plt.title('Stock Market LSTM Forecast')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Create GUI for user input and interaction
def create_gui():
    root = tk.Tk()
    root.title("Stock Market Prediction")

    # Input for stock symbol
    tk.Label(root, text="Enter Stock Symbol (e.g., XXXXXX.ss)").grid(row=0, column=0)
    symbol_entry = tk.Entry(root)
    symbol_entry.grid(row=0, column=1)

    # Input for start date
    tk.Label(root, text="Enter Start Date (YYYY-MM-DD)").grid(row=1, column=0)
    start_date_entry = tk.Entry(root)
    start_date_entry.grid(row=1, column=1)

    # Input for end date
    tk.Label(root, text="Enter End Date (YYYY-MM-DD)").grid(row=2, column=0)
    end_date_entry = tk.Entry(root)
    end_date_entry.grid(row=2, column=1)

    # Progress Bar
    progress = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, variable=progress, maximum=100)
    progress_bar.grid(row=3, column=0, columnspan=2)

    def on_start_button_click():
        symbol = symbol_entry.get()
        start_date = start_date_entry.get()
        end_date = end_date_entry.get()

        # Validate inputs
        if not symbol or not start_date or not end_date:
            messagebox.showerror("Input Error", "All fields are required.")
            return

        # Fetch data
        data = fetch_market_data(symbol, start_date, end_date)
        if data is None:
            messagebox.showerror("Error", "Failed to fetch data. Please check your network connection or try again later.")
            return

        # Data Preprocessing and Model Training
        data_scaled, scaler = preprocess_data(data)
        arima_forecast = arima_predict(data_scaled)
        lstm_forecast = lstm_predict(data_scaled)

        # Perform Analysis using OpenAI
        analysis = analyze_with_deepseek(data)
        print(f"OpenAI analysis result: {analysis}")

        # Show predictions and analysis in window
        show_predictions_in_window(arima_forecast, lstm_forecast, analysis)

        # Plot the predictions
        plot_predictions(data, arima_forecast, lstm_forecast)

    # Start button to trigger prediction
    tk.Button(root, text="Start", command=on_start_button_click).grid(row=4, column=0, columnspan=2)

    root.mainloop()