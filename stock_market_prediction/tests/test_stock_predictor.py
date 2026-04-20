"""
测试重构后的 StockPredictor 类
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from stock_market_prediction.utils import StockPredictor


class TestStockPredictor(unittest.TestCase):
    """测试 StockPredictor 类"""
    
    def setUp(self):
        """设置测试环境"""
        self.predictor = StockPredictor()
        
        # 创建样本数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Open': np.random.rand(100) * 10 + 100,
            'High': np.random.rand(100) * 10 + 105,
            'Low': np.random.rand(100) * 10 + 95,
            'Close': np.random.rand(100) * 10 + 102,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    def test_preprocess_data(self):
        """测试数据预处理"""
        data_scaled, scaler = self.predictor.preprocess_data(self.sample_data)
        
        self.assertIsNotNone(data_scaled)
        self.assertIsNotNone(scaler)
        self.assertEqual(data_scaled.shape[1], 5)  # 5 个特征
        self.assertTrue(np.all(data_scaled >= 0))
        self.assertTrue(np.all(data_scaled <= 1))
    
    def test_preprocess_data_with_missing_values(self):
        """测试含缺失值的数据预处理"""
        data_with_nan = self.sample_data.copy()
        data_with_nan.iloc[5, 2] = np.nan
        data_with_nan.iloc[10, 4] = np.nan
        
        data_scaled, scaler = self.predictor.preprocess_data(data_with_nan)
        
        self.assertFalse(np.isnan(data_scaled).any())
    
    def test_arima_predict(self):
        """测试 ARIMA 预测"""
        data_scaled, _ = self.predictor.preprocess_data(self.sample_data)
        forecast = self.predictor.arima_predict(data_scaled)
        
        self.assertIsNotNone(forecast)
        self.assertEqual(len(forecast), 5)  # 预测 5 天
    
    def test_lstm_predict(self):
        """测试 LSTM 预测"""
        data_scaled, _ = self.predictor.preprocess_data(self.sample_data)
        forecast = self.predictor.lstm_predict(data_scaled, time_step=20)
        
        self.assertIsNotNone(forecast)
        self.assertEqual(len(forecast), 1)
    
    def test_calculate_technical_indicators(self):
        """测试技术指标计算"""
        indicators = self.predictor.calculate_technical_indicators(self.sample_data)
        
        self.assertIn('MA5', indicators)
        self.assertIn('MA10', indicators)
        self.assertIn('MA20', indicators)
        self.assertIn('RSI', indicators)
        self.assertIn('MACD', indicators)
        self.assertIn('Current_Price', indicators)
        self.assertIn('Volume_Ratio', indicators)
    
    def test_fetch_market_data_mock(self):
        """测试数据获取（使用 mock）"""
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = self.sample_data
            
            data = self.predictor.fetch_market_data("AAPL", "2023-01-01", "2023-04-10")
            
            self.assertIsNotNone(data)
            self.assertFalse(data.empty)
    
    def test_run_full_analysis(self):
        """测试完整分析流程（使用 mock）"""
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = self.sample_data
            
            results = self.predictor.run_full_analysis(
                symbol="AAPL",
                start_date="2023-01-01",
                end_date="2023-04-10"
            )
            
            self.assertTrue(results['success'])
            self.assertIsNone(results['error'])
            self.assertIsNotNone(results['data'])
            self.assertIsNotNone(results['arima_forecast'])
            self.assertIsNotNone(results['lstm_forecast'])
            self.assertIsNotNone(results['technical_indicators'])


if __name__ == '__main__':
    unittest.main()
