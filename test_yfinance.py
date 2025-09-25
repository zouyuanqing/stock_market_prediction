import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'stock_market_prediction'))

from stock_market_prediction.utils import fetch_market_data
import pandas as pd

def test_yfinance_with_proxy():
    """测试使用代理获取数据"""
    print("Testing Yahoo Finance data fetching with enhanced retry mechanism...")
    
    # 尝试获取数据，使用增强的重试机制
    data = fetch_market_data("AAPL", "2020-01-01", "2020-01-31", max_retries=5, retry_delay=10)
    
    if data is not None and not data.empty:
        print("Successfully fetched data:")
        print(data.head())
        print(f"Data shape: {data.shape}")
        return True
    else:
        print("Failed to fetch data after all retries")
        return False

def test_yfinance_with_different_symbol():
    """测试获取不同的股票数据"""
    print("\nTesting with a different symbol...")
    
    # 尝试获取微软的数据
    data = fetch_market_data("MSFT", "2020-01-01", "2020-01-31", max_retries=3, retry_delay=5)
    
    if data is not None and not data.empty:
        print("Successfully fetched MSFT data:")
        print(data.head())
        return True
    else:
        print("Failed to fetch MSFT data")
        return False

if __name__ == "__main__":
    print("Running Yahoo Finance tests...")
    
    # 测试苹果数据
    success1 = test_yfinance_with_proxy()
    
    # 测试微软数据
    success2 = test_yfinance_with_different_symbol()
    
    if success1 or success2:
        print("\nAt least one test succeeded!")
    else:
        print("\nAll tests failed. You might need to use a proxy or wait before trying again.")