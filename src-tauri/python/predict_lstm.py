#!/usr/bin/env python3
"""
LSTM prediction script (multi-step) - communicates with Rust via stdin/stdout.
Predicts the next `steps` closing prices using iterative (recursive) multi-step
forecasting: each predicted value is fed back into the sliding window.
"""

import sys
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def main():
    try:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import MinMaxScaler

        # Reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Read stdin
        input_data = sys.stdin.read()
        data = json.loads(input_data)

        prices = data.get("prices", [])
        window_size = data.get("window_size", 20)
        epochs = data.get("epochs", 100)
        hidden_size = data.get("hidden_size", 64)
        num_layers = data.get("num_layers", 2)
        steps = int(data.get("steps", 5))

        # Validate data
        if len(prices) < 60:
            result = {"error": "insufficient data, need >= 60 points"}
            print(json.dumps(result))
            return

        prices = np.array(prices, dtype=np.float32).reshape(-1, 1)

        # More epochs for smaller datasets
        if len(prices) < 100:
            epochs = max(epochs, 200)

        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices)

        # Build sliding-window samples
        X, y = [], []
        for i in range(window_size, len(prices_scaled)):
            X.append(prices_scaled[i - window_size : i])
            y.append(prices_scaled[i])

        X = np.array(X)
        y = np.array(y)

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # 80/20 split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

        # Define LSTM model
        class LSTMPredictor(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.2 if num_layers > 1 else 0,
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Take the output of the last time step
                last_output = lstm_out[:, -1, :]
                prediction = self.fc(last_output)
                return prediction

        model = LSTMPredictor(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train
        train_loss = 0.0
        val_loss = 0.0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output, y_val).item()

        # Multi-step prediction: feed predictions back iteratively
        model.eval()
        with torch.no_grad():
            current = prices_scaled[-window_size:].reshape(1, window_size, 1)
            current_tensor = torch.FloatTensor(current)
            preds_scaled = []
            for _ in range(steps):
                pred = model(current_tensor)  # shape (1, 1)
                preds_scaled.append(float(pred.item()))
                pred_reshaped = pred.reshape(1, 1, 1)
                current_tensor = torch.cat(
                    [current_tensor[:, 1:, :], pred_reshaped], dim=1
                )

            predictions = scaler.inverse_transform(
                np.array(preds_scaled).reshape(-1, 1)
            ).reshape(-1).tolist()
            predictions = [max(0.0, round(p, 2)) for p in predictions]

        result = {
            "predictions": predictions,
            "prediction": predictions[0],
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
        }

        print(json.dumps(result))

    except ImportError as e:
        result = {"error": f"Missing Python dependency: {e}. Please run: pip install torch scikit-learn numpy"}
        print(json.dumps(result))
    except Exception as e:
        result = {"error": str(e)}
        print(json.dumps(result))


if __name__ == "__main__":
    main()
