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
import json
from dotenv import load_dotenv
import markdown
from tkinter import font

# 设置OpenAI API密钥
def setup_openai_api():
    """设置OpenAI API配置"""
    # 首先尝试从环境变量加载
    load_dotenv()
    
    # 获取API配置
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    
    if api_key:
        openai.api_base = api_base
        openai.api_key = api_key
        print(f"DeepSeek API已配置，使用URL: {api_base}")
        return True
    else:
        print("注意：未设置DeepSeek API密钥，AI分析功能将不可用。")
        print("如需使用AI分析功能，请在.env文件中设置DEEPSEEK_API_KEY。")
        return False

# 初始化API配置
api_configured = setup_openai_api()

# Fetch market data (10 years of data) from multiple sources
def fetch_market_data(symbol, start_date, end_date, max_retries=5, retry_delay=10, proxies=None, data_source='yahoo'):
    """
    从多个数据源获取股票数据，包含重试机制、代理支持和错误处理
    
    注意：Yahoo Finance 和 Stooq 在某些地区可能存在访问限制，
    如果遇到连接问题，建议使用代理或配置其他数据源。
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        max_retries: 最大重试次数
        retry_delay: 重试延迟时间（秒）
        proxies: 代理设置
        data_source: 数据源 ('yahoo', 'stooq')
    
    Returns:
        DataFrame: 股票数据
    """
    # 根据选择的数据源获取数据
    if data_source == 'yahoo':
        print(f"Fetching data for {symbol} from Yahoo Finance...")
        return fetch_from_yahoo_finance(symbol, start_date, end_date, max_retries, retry_delay, proxies)
    elif data_source == 'stooq':
        print(f"Fetching data for {symbol} from Stooq...")
        return fetch_from_stooq(symbol, start_date, end_date, max_retries, retry_delay, proxies)
    else:
        # 默认使用Yahoo Finance
        print(f"Fetching data for {symbol} from default source (Yahoo Finance)...")
        return fetch_from_yahoo_finance(symbol, start_date, end_date, max_retries, retry_delay, proxies)

def fetch_from_yahoo_finance(symbol, start_date, end_date, max_retries=5, retry_delay=10, proxies=None):
    """
    从Yahoo Finance获取股票数据
    """
    # 设置代理（如果提供）
    if proxies:
        try:
            yf.pdr_override()
            import yfinance.shared as yfs
            yfs.session.proxies = proxies
        except:
            pass  # 如果代理设置失败，继续尝试不使用代理
    
    for attempt in range(max_retries):
        try:
            # 添加随机延迟避免频率限制
            if attempt > 0:
                # 对于频率限制错误，增加更长的延迟时间
                delay = retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 5)
                print(f"Waiting {delay:.2f} seconds before retry...")
                time.sleep(delay)
            
            # 设置请求参数
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data is not None and not data.empty:
                print(f"Successfully fetched data for {symbol} from Yahoo Finance")
                return data
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {symbol}: {e}")
            # 特别处理频率限制错误
            if "RateLimit" in str(e) or "Too Many Requests" in str(e):
                print("Rate limit error detected. Increasing wait time...")
            
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
            else:
                print(f"Failed to fetch data for {symbol} from Yahoo Finance after {max_retries} attempts")
    
    return None

def fetch_from_stooq(symbol, start_date, end_date, max_retries=3, retry_delay=5, proxies=None):
    """
    从Stooq获取股票数据（免费且无频率限制）
    """
    try:
        import pandas_datareader as pdr
        from pandas_datareader import data as web
        
        # 调整股票代码格式以适应Stooq
        formatted_symbol = format_symbol_for_stooq(symbol)
        
        # Stooq通常不需要重试机制，因为它没有严格的频率限制
        data = web.DataReader(formatted_symbol, 'stooq', start_date, end_date)
        
        if data is not None and not data.empty:
            print(f"Successfully fetched data for {symbol} from Stooq")
            # 确保列名一致
            data = standardize_dataframe_columns(data)
            return data
        else:
            print(f"No data returned from Stooq for {symbol}")
            return None
            
    except ImportError:
        print("pandas_datareader not installed")
        return None
    except Exception as e:
        print(f"Error fetching data from Stooq: {e}")
        return None

def fetch_from_alpha_vantage(symbol, start_date, end_date, max_retries=3, retry_delay=5, proxies=None):
    """
    从Alpha Vantage获取股票数据（需要API密钥，但有免费套餐）
    """
    try:
        # Alpha Vantage提供免费API密钥，每天5次请求
        # 这里我们只在环境变量中提供了API密钥时才使用
        api_key = os.getenv('ALPHAVANTAGE_API_KEY')
        if not api_key:
            print("Alpha Vantage API key not set, skipping this data source")
            return None
        
        import pandas_datareader as pdr
        from pandas_datareader import data as web
        
        # Alpha Vantage通常不需要重试机制
        data = web.DataReader(symbol, 'av-daily', start_date, end_date, api_key=api_key)
        
        if data is not None and not data.empty:
            print(f"Successfully fetched data for {symbol} from Alpha Vantage")
            # 确保列名一致
            data = standardize_dataframe_columns(data)
            return data
        else:
            print(f"No data returned from Alpha Vantage for {symbol}")
            return None
            
    except ImportError:
        print("pandas_datareader not installed")
        return None
    except Exception as e:
        print(f"Error fetching data from Alpha Vantage: {e}")
        return None

def format_symbol_for_stooq(symbol):
    """
    根据Stooq要求格式化股票代码
    """
    # Stooq使用不同的符号格式
    if '.SS' in symbol:
        return symbol.replace('.SS', '.SH')  # 上海证券交易所
    elif '.SZ' in symbol:
        return symbol.replace('.SZ', '.SZ')  # 深圳证券交易所保持不变
    elif symbol.endswith('.L'):
        return symbol.replace('.L', '.LON')  # 伦敦证券交易所
    elif symbol.endswith('.F'):
        return symbol.replace('.F', '.FRA')  # 法兰克福证券交易所
    elif symbol.endswith('.DE'):
        return symbol.replace('.DE', '.GER')  # 德国证券交易所
    elif symbol.endswith('.PA'):
        return symbol.replace('.PA', '.PAR')  # 巴黎证券交易所
    elif symbol.endswith('.AS'):
        return symbol.replace('.AS', '.AMS')  # 阿姆斯特丹证券交易所
    elif symbol.endswith('.BR'):
        return symbol.replace('.BR', '.BRU')  # 布鲁塞尔证券交易所
    elif symbol.endswith('.TO'):
        return symbol  # 多伦多证券交易所保持不变
    elif symbol.endswith('.V'):
        return symbol.replace('.V', '.VAN')  # 温哥华证券交易所
    elif symbol.endswith('.AX'):
        return symbol  # 澳大利亚证券交易所保持不变
    elif symbol.endswith('.JK'):
        return symbol.replace('.JK', '.JKT')  # 雅加达证券交易所
    elif symbol.endswith('.KS'):
        return symbol.replace('.KS', '.KRX')  # 韩国证券交易所
    elif symbol.endswith('.T'):
        return symbol.replace('.T', '.TYO')  # 东京证券交易所
    
    return symbol

def standardize_dataframe_columns(df):
    """
    标准化DataFrame列名以匹配预期格式
    """
    # 创建列名映射
    column_mapping = {}
    for col in df.columns:
        lower_col = col.lower()
        if 'open' in lower_col:
            column_mapping[col] = 'Open'
        elif 'high' in lower_col:
            column_mapping[col] = 'High'
        elif 'low' in lower_col:
            column_mapping[col] = 'Low'
        elif 'close' in lower_col:
            column_mapping[col] = 'Close'
        elif 'volume' in lower_col:
            column_mapping[col] = 'Volume'
        else:
            column_mapping[col] = col  # 保持其他列名不变
    
    # 重命名列
    df = df.rename(columns=column_mapping)
    
    # 确保必要的列存在
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            # 如果缺少某列，设置为0
            df[col] = 0
    
    # 只返回需要的列
    return df[required_columns]

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
    
    # Check if we have enough data
    if len(X) == 0:
        # Return a default forecast if not enough data
        return np.array([[0.5]])
    
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
    # 修复LSTM预测结果的显示格式
    try:
        if isinstance(lstm_forecast, np.ndarray):
            # 如果是多维数组，提取第一个元素
            if lstm_forecast.ndim > 0:
                forecast_value = lstm_forecast.flat[0]
            else:
                forecast_value = float(lstm_forecast)
        else:
            forecast_value = float(lstm_forecast)
        lstm_text.insert(tk.END, f"LSTM 预测股价: {forecast_value:.2f}\n")
    except Exception as e:
        lstm_text.insert(tk.END, f"LSTM 预测结果格式化错误: {str(e)}\n")
        # 显示原始预测结果
        lstm_text.insert(tk.END, f"原始预测结果: {lstm_forecast}\n")
    lstm_text.config(state=tk.DISABLED)

    # Analysis Results Tab with Markdown Rendering
    analysis_frame = ttk.Frame(notebook)
    notebook.add(analysis_frame, text="AI 深度分析")
    
    # 创建文本控件用于显示分析结果
    analysis_text = tk.Text(analysis_frame, wrap=tk.WORD)
    analysis_text.pack(fill='both', expand=True)
    
    if analysis:
        # 将markdown转换为格式化文本
        try:
            formatted_analysis = convert_markdown_to_text(analysis)
            analysis_text.insert(tk.END, formatted_analysis)
            
            # 为标题添加特殊格式
            format_text_widget(analysis_text)
        except Exception as e:
            # 如果转换失败，直接显示原始文本
            print(f"Markdown转换失败: {e}")
            analysis_text.insert(tk.END, analysis)
    else:
        analysis_text.insert(tk.END, "暂无AI分析结果（DeepSeek API密钥未设置或请求失败）")
    analysis_text.config(state=tk.DISABLED)

    # Add scrollbars
    for text_widget in [arima_text, lstm_text, analysis_text]:
        scrollbar = tk.Scrollbar(text_widget, orient="vertical", command=text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        text_widget.config(yscrollcommand=scrollbar.set)

    root.mainloop()

def convert_markdown_to_text(markdown_text):
    """
    简单的Markdown到格式化文本的转换函数
    """
    # 创建标签样式
    formatted_text = ""
    
    # 按行处理
    lines = markdown_text.split('\n')
    
    for line in lines:
        stripped_line = line.strip()
        
        # 处理标题
        if stripped_line.startswith('# '):
            formatted_text += f"\n=== {stripped_line[2:]} ===\n\n"
        elif stripped_line.startswith('## '):
            formatted_text += f"\n--- {stripped_line[3:]} ---\n\n"
        elif stripped_line.startswith('### '):
            formatted_text += f"\n{stripped_line[4:]}:\n"
        # 处理粗体
        elif '**' in stripped_line or '__' in stripped_line:
            # 简单替换粗体标记
            processed_line = stripped_line.replace('**', '*').replace('__', '*')
            formatted_text += processed_line + "\n"
        # 处理列表
        elif stripped_line.startswith('- ') or stripped_line.startswith('* '):
            formatted_text += f"  • {stripped_line[2:]}\n"
        # 处理数字列表
        elif stripped_line.startswith('1. ') or stripped_line.startswith('2. ') or stripped_line.startswith('3. '):
            formatted_text += f"  {stripped_line}\n"
        # 处理空行
        elif stripped_line == "":
            formatted_text += "\n"
        # 普通文本
        else:
            formatted_text += stripped_line + "\n"
    
    return formatted_text

def format_text_widget(text_widget):
    """
    为文本控件中的特定内容添加格式
    """
    content = text_widget.get("1.0", tk.END)
    
    # 为标题添加格式
    start = "1.0"
    while True:
        pos = text_widget.search("=== ", start, tk.END)
        if not pos:
            break
        end_pos = text_widget.search(" ===", pos, tk.END)
        if end_pos:
            # 设置标题格式
            text_widget.tag_add("header1", pos, f"{end_pos}+4c")
            start = f"{end_pos}+4c"
        else:
            break
    
    # 为子标题添加格式
    start = "1.0"
    while True:
        pos = text_widget.search("--- ", start, tk.END)
        if not pos:
            break
        end_pos = text_widget.search(" ---", pos, tk.END)
        if end_pos:
            # 设置子标题格式
            text_widget.tag_add("header2", pos, f"{end_pos}+4c")
            start = f"{end_pos}+4c"
        else:
            break
    
    # 配置标签样式
    text_widget.tag_config("header1", font=("Arial", 12, "bold"), foreground="blue")
    text_widget.tag_config("header2", font=("Arial", 11, "bold"), foreground="darkblue")

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

def save_api_settings_to_env(env_file_path, api_base, api_key):
    """
    保存API设置到.env文件
    
    Args:
        env_file_path (str): .env文件路径
        api_base (str): API Base URL
        api_key (str): API密钥
    """
    # 读取现有的.env文件内容
    env_lines = []
    if os.path.exists(env_file_path):
        with open(env_file_path, 'r', encoding='utf-8') as f:
            env_lines = f.readlines()
    
    # 更新或添加API设置
    new_env_lines = []
    base_found = False
    key_found = False
    
    for line in env_lines:
        if line.startswith('DEEPSEEK_API_BASE='):
            new_env_lines.append(f'DEEPSEEK_API_BASE={api_base}\n')
            base_found = True
        elif line.startswith('DEEPSEEK_API_KEY='):
            new_env_lines.append(f'DEEPSEEK_API_KEY={api_key}\n')
            key_found = True
        else:
            new_env_lines.append(line)
    
    # 如果没有找到相应的设置，则添加它们
    if not base_found:
        # 找到合适的位置插入API设置
        insert_pos = len(new_env_lines)
        for i, line in enumerate(new_env_lines):
            if line.startswith('# DeepSeek API配置'):
                insert_pos = i + 1
                break
        new_env_lines.insert(insert_pos, f'DEEPSEEK_API_BASE={api_base}\n')
        
    if not key_found:
        # 找到合适的位置插入API密钥设置
        insert_pos = len(new_env_lines)
        for i, line in enumerate(new_env_lines):
            if line.startswith('# DeepSeek API配置'):
                insert_pos = i + 1
                break
        new_env_lines.insert(insert_pos, f'DEEPSEEK_API_KEY={api_key}\n')
    
    # 写入更新后的内容到.env文件
    with open(env_file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_env_lines)

# Create GUI for user input and interaction
def create_gui():
    root = tk.Tk()
    root.title("股票市场预测工具")
    root.geometry("550x500")  # 增加高度以容纳API设置

    # Add padding to the main window
    main_frame = ttk.Frame(root, padding="20")
    main_frame.grid(row=0, column=0, sticky="WENS")

    # Configure grid weights for responsive design
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)

    # API设置区域
    api_frame = ttk.LabelFrame(main_frame, text="API设置（可选）", padding="10")
    api_frame.grid(row=0, column=0, columnspan=2, sticky="WE", pady=(0, 10))
    api_frame.columnconfigure(1, weight=1)

    # API Base URL
    ttk.Label(api_frame, text="API Base URL:").grid(row=0, column=0, sticky=tk.W, pady=2)
    api_base_var = tk.StringVar(value=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    api_base_entry = ttk.Entry(api_frame, textvariable=api_base_var, width=40)
    api_base_entry.grid(row=0, column=1, sticky="WE", pady=2, padx=(5, 0))

    # API Key
    ttk.Label(api_frame, text="API Key:").grid(row=1, column=0, sticky=tk.W, pady=2)
    api_key_var = tk.StringVar(value=os.getenv("DEEPSEEK_API_KEY", ""))
    api_key_entry = ttk.Entry(api_frame, textvariable=api_key_var, width=40, show="*")
    api_key_entry.grid(row=1, column=1, sticky="WE", pady=2, padx=(5, 0))

    # 保存API设置按钮
    def save_api_settings():
        api_base = api_base_var.get().strip()
        api_key = api_key_var.get().strip()
        
        # 获取.env文件路径
        env_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
        
        # 保存API设置到.env文件
        save_api_settings_to_env(env_file_path, api_base, api_key)
        
        messagebox.showinfo("保存成功", "API设置已保存到.env文件中！")
        
        # 如果提供了API密钥，则更新OpenAI配置
        if api_key:
            openai.api_base = api_base
            openai.api_key = api_key
            print(f"已更新API配置: {api_base}")

    save_api_button = ttk.Button(api_frame, text="保存API设置", command=save_api_settings)
    save_api_button.grid(row=2, column=1, sticky="E", pady=5)

    # Input for stock symbol
    ttk.Label(main_frame, text="股票代码 (例如: AAPL, 000001.ss):").grid(row=1, column=0, sticky=tk.W, pady=5)
    symbol_entry = ttk.Entry(main_frame, width=30)
    symbol_entry.grid(row=1, column=1, sticky="WE", pady=5)

    # Input for start date
    ttk.Label(main_frame, text="开始日期 (YYYY-MM-DD):").grid(row=2, column=0, sticky=tk.W, pady=5)
    start_date_entry = ttk.Entry(main_frame, width=30)
    start_date_entry.grid(row=2, column=1, sticky="WE", pady=5)
    start_date_entry.insert(0, "2020-01-01")  # Default value

    # Input for end date
    ttk.Label(main_frame, text="结束日期 (YYYY-MM-DD):").grid(row=3, column=0, sticky=tk.W, pady=5)
    end_date_entry = ttk.Entry(main_frame, width=30)
    end_date_entry.grid(row=3, column=1, sticky="WE", pady=5)
    end_date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))  # Default to today

    # Data source selection
    ttk.Label(main_frame, text="数据源:").grid(row=4, column=0, sticky=tk.W, pady=5)
    data_source_var = tk.StringVar(value="yahoo")  # Default to Yahoo Finance
    data_source_combo = ttk.Combobox(main_frame, textvariable=data_source_var, 
                                    values=["yahoo", "stooq"],  # 目前支持的数据源
                                    state="readonly", width=27)
    data_source_combo.grid(row=4, column=1, sticky="WE", pady=5)

    # Progress Bar
    progress = tk.DoubleVar()
    progress_bar = ttk.Progressbar(main_frame, variable=progress, maximum=100, length=300)
    progress_bar.grid(row=5, column=0, columnspan=2, pady=10, sticky="WE")

    # Status Label
    status_label = ttk.Label(main_frame, text="准备就绪")
    status_label.grid(row=6, column=0, columnspan=2, pady=5)

    def update_status(text):
        status_label.config(text=text)
        root.update_idletasks()

    def on_start_button_click():
        # 获取API设置
        api_base = api_base_var.get().strip()
        api_key = api_key_var.get().strip()
        
        # 如果提供了API密钥，则更新OpenAI配置
        if api_key:
            openai.api_base = api_base
            openai.api_key = api_key
            print(f"已更新API配置: {api_base}")
        
        symbol = symbol_entry.get().strip()
        start_date = start_date_entry.get().strip()
        end_date = end_date_entry.get().strip()
        data_source = data_source_var.get()  # 获取选择的数据源

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

        # Fetch data with selected data source
        data = fetch_market_data(symbol, start_date, end_date, data_source=data_source)
        if data is None:
            messagebox.showerror("数据获取失败", 
                               f"无法获取股票数据。可能的原因：\n"
                               f"1. 网络连接问题\n"
                               f"2. 股票代码不正确\n"
                               f"3. 请求频率过高，请稍后再试\n"
                               f"4. 数据源 '{data_source}' 暂时不可用\n"
                               f"5. 尝试切换到其他数据源")
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
    start_button.grid(row=7, column=0, columnspan=2, pady=20)

    # Add tooltips (removed due to compatibility issues)

    root.mainloop()