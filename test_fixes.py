import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'stock_market_prediction'))

from stock_market_prediction.utils import show_predictions_in_window

def test_lstm_display():
    """测试LSTM预测结果的显示"""
    print("Testing LSTM prediction display...")
    
    # 测试不同格式的LSTM预测结果
    test_cases = [
        np.array([[0.75]]),  # 二维数组
        np.array([0.75]),    # 一维数组
        np.array(0.75),      # 标量
        0.75,                # 浮点数
    ]
    
    for i, forecast in enumerate(test_cases):
        print(f"\nTest case {i+1}: {type(forecast)}")
        print(f"Value: {forecast}")
        
        try:
            # 模拟显示函数中的处理逻辑
            if isinstance(forecast, np.ndarray):
                if forecast.ndim > 0:
                    forecast_value = forecast.flat[0]
                else:
                    forecast_value = float(forecast)
            else:
                forecast_value = float(forecast)
            
            print(f"Formatted value: {forecast_value:.2f}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_lstm_display()