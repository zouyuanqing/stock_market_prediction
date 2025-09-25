import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'stock_market_prediction'))

from stock_market_prediction.utils import fetch_market_data
import pandas as pd

def test_stooq():
    """测试Stooq数据源"""
    print("Testing Stooq data source...")
    
    # 测试一些国际股票
    test_symbols = [
        "AAPL.US",  # 苹果 - 美国
        "BMW.DE",   # 宝马 - 德国
        "NESN.SW",  # 雀巢 - 瑞士
    ]
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol} from Stooq...")
        try:
            data = fetch_market_data(symbol, "2020-01-01", "2020-01-31", data_source='stooq')
            if data is not None and not data.empty:
                print(f"Successfully fetched {symbol} data from Stooq:")
                print(data.head())
                print(f"Data shape: {data.shape}")
            else:
                print(f"No data returned for {symbol} from Stooq")
        except Exception as e:
            print(f"Error fetching {symbol} from Stooq: {e}")

if __name__ == "__main__":
    test_stooq()