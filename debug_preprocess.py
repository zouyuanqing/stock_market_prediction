import pandas as pd
import numpy as np
from stock_market_prediction.utils import preprocess_data

# Create sample data for testing with more realistic values
dates = pd.date_range('2020-01-01', periods=30, freq='D')
# Create more realistic stock data
open_prices = np.random.rand(30) * 50 + 100  # Open prices between 100-150
high_prices = open_prices + np.random.rand(30) * 5  # High slightly above open
low_prices = open_prices - np.random.rand(30) * 5   # Low slightly below open
close_prices = open_prices + np.random.rand(30) * 10 - 5  # Close around open
volumes = np.random.randint(1000000, 10000000, 30)  # Volumes between 1M-10M

sample_data = pd.DataFrame({
    'Open': open_prices,
    'High': high_prices,
    'Low': low_prices,
    'Close': close_prices,
    'Volume': volumes
}, index=dates)

print("Original data:")
print(sample_data.head())
print("\nData statistics:")
print(sample_data.describe())

# Preprocess the data
data_scaled, scaler = preprocess_data(sample_data)

print("\nScaled data statistics:")
print(f"Min values: {np.min(data_scaled, axis=0)}")
print(f"Max values: {np.max(data_scaled, axis=0)}")
print(f"All values <= 1: {np.all(data_scaled <= 1)}")
print(f"All values >= 0: {np.all(data_scaled >= 0)}")

# Check for any values outside [0, 1] range
outside_range = data_scaled[(data_scaled < 0) | (data_scaled > 1)]
if len(outside_range) > 0:
    print(f"\nValues outside [0, 1] range: {outside_range}")
else:
    print("\nAll values are within [0, 1] range")