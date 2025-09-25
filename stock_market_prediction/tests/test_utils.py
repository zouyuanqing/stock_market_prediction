import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from stock_market_prediction.utils import fetch_market_data, preprocess_data, arima_predict, lstm_predict

class TestStockMarketPrediction(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create sample data for testing
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        self.sample_data = pd.DataFrame({
            'Open': np.random.rand(30) * 100 + 100,
            'High': np.random.rand(30) * 100 + 110,
            'Low': np.random.rand(30) * 100 + 90,
            'Close': np.random.rand(30) * 100 + 105,
            'Volume': np.random.randint(1000000, 10000000, 30)
        }, index=dates)
    
    def test_fetch_market_data_success(self):
        """Test successful data fetching"""
        # Mock yf.download to return sample data
        with patch('stock_market_prediction.utils.yf.download') as mock_download:
            mock_download.return_value = self.sample_data
            data = fetch_market_data("AAPL", "2020-01-01", "2020-01-31")
            self.assertIsNotNone(data)
            self.assertFalse(data.empty)
            self.assertEqual(len(data), 30)
    
    def test_fetch_market_data_empty_result(self):
        """Test handling of empty data result"""
        # Mock yf.download to return empty DataFrame
        with patch('stock_market_prediction.utils.yf.download') as mock_download:
            mock_download.return_value = pd.DataFrame()
            data = fetch_market_data("INVALID", "2020-01-01", "2020-01-31")
            self.assertIsNone(data)
    
    def test_fetch_market_data_with_exception(self):
        """Test handling of exceptions during data fetching"""
        # Mock yf.download to raise an exception
        with patch('stock_market_prediction.utils.yf.download') as mock_download:
            mock_download.side_effect = Exception("Network error")
            data = fetch_market_data("AAPL", "2020-01-01", "2020-01-31")
            self.assertIsNone(data)
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        data_scaled, scaler = preprocess_data(self.sample_data)
        self.assertIsNotNone(data_scaled)
        self.assertIsNotNone(scaler)
        # Check that scaled data is in range [0, 1]
        self.assertTrue(np.all(data_scaled >= 0))
        self.assertTrue(np.all(data_scaled <= 1))
        # Check shape - should have 5 features (Open, High, Low, Close, Volume)
        self.assertEqual(data_scaled.shape[1], 5)
    
    def test_preprocess_data_missing_values(self):
        """Test preprocessing with missing values"""
        # Create data with some missing values
        data_with_nan = self.sample_data.copy()
        data_with_nan.iloc[5, 2] = np.nan  # Set one value to NaN
        data_with_nan.iloc[10, 4] = np.nan  # Set another value to NaN
        
        data_scaled, scaler = preprocess_data(data_with_nan)
        self.assertIsNotNone(data_scaled)
        self.assertIsNotNone(scaler)
        # Check that there are no NaN values after preprocessing
        self.assertFalse(np.isnan(data_scaled).any())
    
    def test_arima_predict(self):
        """Test ARIMA prediction"""
        data_scaled, _ = preprocess_data(self.sample_data)
        forecast = arima_predict(data_scaled)
        self.assertIsNotNone(forecast)
        # Check that we get 5 forecasted values
        self.assertEqual(len(forecast), 5)
    
    def test_lstm_predict(self):
        """Test LSTM prediction"""
        data_scaled, _ = preprocess_data(self.sample_data)
        forecast = lstm_predict(data_scaled)
        self.assertIsNotNone(forecast)
        # Check that we get a forecasted value
        self.assertTrue(hasattr(forecast, '__len__'))

if __name__ == "__main__":
    unittest.main()