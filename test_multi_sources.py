import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'stock_market_prediction'))

from stock_market_prediction.utils import fetch_market_data
import pandas as pd

def test_multi_sources():
    """测试多数据源获取数据"""
    print("Testing multi-source data fetching...")
    
    # 测试苹果数据
    print("\n1. Testing AAPL (Apple)...")
    data = fetch_market_data("AAPL", "2020-01-01", "2020-01-31")
    if data is not None and not data.empty:
        print("Successfully fetched AAPL data:")
        print(data.head())
        print(f"Data shape: {data.shape}")
    else:
        print("Failed to fetch AAPL data from all sources")
    
    # 测试微软数据
    print("\n2. Testing MSFT (Microsoft)...")
    data = fetch_market_data("MSFT", "2020-01-01", "2020-01-31")
    if data is not None and not data.empty:
        print("Successfully fetched MSFT data:")
        print(data.head())
        print(f"Data shape: {data.shape}")
    else:
        print("Failed to fetch MSFT data from all sources")
    
    # 测试谷歌数据
    print("\n3. Testing GOOGL (Google)...")
    data = fetch_market_data("GOOGL", "2020-01-01", "2020-01-31")
    if data is not None and not data.empty:
        print("Successfully fetched GOOGL data:")
        print(data.head())
        print(f"Data shape: {data.shape}")
    else:
        print("Failed to fetch GOOGL data from all sources")

def test_stooq_source():
    """测试Stooq数据源"""
    print("\n\nTesting Stooq data source specifically...")
    
    # 测试一些国际股票
    test_symbols = [
        "AAPL.US",  # 苹果 - 美国
        "BMW.DE",   # 宝马 - 德国
        "NESN.SW",  # 雀巢 - 瑞士
    ]
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}...")
        try:
            import pandas_datareader as pdr
            from pandas_datareader import data as web
            
            # 尝试直接从Stooq获取数据
            data = web.DataReader(symbol, 'stooq', "2020-01-01", "2020-01-31")
            if data is not None and not data.empty:
                print(f"Successfully fetched {symbol} data from Stooq:")
                print(data.head())
                print(f"Data shape: {data.shape}")
            else:
                print(f"No data returned for {symbol} from Stooq")
        except Exception as e:
            print(f"Error fetching {symbol} from Stooq: {e}")

if __name__ == "__main__":
    print("Running multi-source data fetching tests...")
    test_multi_sources()
    test_stooq_source()
    print("\nTests completed.")