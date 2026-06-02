#!/usr/bin/env python3
"""
LSTM 预测脚本 - 通过 stdin/stdout 与 Rust 通信
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

        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)

        # 读取 stdin
        input_data = sys.stdin.read()
        data = json.loads(input_data)

        prices = data.get("prices", [])
        window_size = data.get("window_size", 20)
        epochs = data.get("epochs", 100)
        hidden_size = data.get("hidden_size", 64)
        num_layers = data.get("num_layers", 2)

        # 检查数据量
        if len(prices) < 60:
            result = {"error": "insufficient data, need >= 60 points"}
            print(json.dumps(result))
            return

        prices = np.array(prices, dtype=np.float32).reshape(-1, 1)

        # 数据量不足时增加 epochs
        if len(prices) < 100:
            epochs = max(epochs, 200)

        # 归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices)

        # 构造滑动窗口样本
        X, y = [], []
        for i in range(window_size, len(prices_scaled)):
            X.append(prices_scaled[i - window_size : i])
            y.append(prices_scaled[i])

        X = np.array(X)
        y = np.array(y)

        # 转换为 PyTorch 张量
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # 80/20 分割
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

        # 定义 LSTM 模型
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
                # 取最后一个时间步的输出
                last_output = lstm_out[:, -1, :]
                prediction = self.fc(last_output)
                return prediction

        # 初始化模型
        model = LSTMPredictor(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers
        )

        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 训练
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

            # 验证
            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output, y_val).item()

        # 预测：使用最后 window_size 个价格
        model.eval()
        with torch.no_grad():
            last_window = prices_scaled[-window_size:].reshape(1, window_size, 1)
            last_window_tensor = torch.FloatTensor(last_window)
            prediction_scaled = model(last_window_tensor).numpy()

            # 反归一化
            prediction = scaler.inverse_transform(prediction_scaled)
            prediction_value = float(prediction[0, 0])

        # 确保预测值合理
        prediction_value = max(0.0, round(prediction_value, 2))

        result = {
            "prediction": prediction_value,
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
