import unittest
from stock_market_prediction.utils import fetch_market_data, preprocess_data

class TestStockMarketPrediction(unittest.TestCase):
    def test_fetch_market_data(self):
        data = fetch_market_data("AAPL", "2020-01-01", "2020-01-31")
        self.assertIsNotNone(data)
        self.assertFalse(data.empty)

    def test_preprocess_data(self):
        data = fetch_market_data("AAPL", "2020-01-01", "2020-01-31")
        data_scaled, scaler = preprocess_data(data)
        self.assertIsNotNone(data_scaled)
        self.assertIsNotNone(scaler)

if __name__ == "__main__":
    unittest.main()
