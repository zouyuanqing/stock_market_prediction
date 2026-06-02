import yfinance as yf
import time
import random
import pandas as pd

def test_yfinance_direct():
    """直接测试yfinance库"""
    print("Directly testing yfinance library...")
    
    try:
        # 先等待一段时间，避免频率限制
        print("Waiting 15 seconds before making request...")
        time.sleep(15)
        
        # 尝试获取苹果的数据
        print("Attempting to fetch AAPL data...")
        data = yf.download("AAPL", start="2020-01-01", end="2020-01-31", progress=False)
        
        if data is None or data.empty:
            print("No data returned")
            return False
        else:
            print("Successfully fetched data:")
            print(data.head())
            print(f"Data shape: {data.shape}")
            return True
            
    except Exception as e:
        print(f"Error occurred: {e}")
        return False

if __name__ == "__main__":
    success = test_yfinance_direct()
    if success:
        print("\nDirect test succeeded!")
    else:
        print("\nDirect test failed. You might need to use a proxy or wait longer before trying again.")