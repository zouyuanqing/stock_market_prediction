#!/usr/bin/env python3
"""
ARIMA 预测脚本 - 通过 stdin/stdout 与 Rust 通信
"""

import sys
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def adf_test(prices):
    """ADF 检验判断平稳性"""
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(prices, autolag="AIC")
        p_value = result[1]
        return p_value < 0.05  # p < 0.05 拒绝原假设，序列平稳
    except Exception:
        return False


def fit_arima_pmdarima(prices, days):
    """使用 pmdarima 自动选择 ARIMA 阶数"""
    import pmdarima as pm

    # 自动搜索最佳阶数
    model = pm.auto_arima(
        prices,
        start_p=0,
        max_p=3,
        d=None,  # 自动判断差分阶数
        max_d=2,
        start_q=0,
        max_q=3,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )

    predictions = model.predict(n_periods=days)
    order = model.order
    aic = model.aic()

    return predictions.tolist(), list(order), aic


def fit_arima_statsmodels(prices, days):
    """使用 statsmodels 手动尝试多组阶数，选 AIC 最小的"""
    from statsmodels.tsa.arima.model import ARIMA

    # 候选阶数
    orders = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2), (3, 1, 1), (1, 1, 3)]

    # 如果序列平稳，尝试 d=0
    if adf_test(prices):
        orders.extend([(1, 0, 1), (2, 0, 1), (1, 0, 2)])

    best_aic = float("inf")
    best_model = None
    best_order = (1, 1, 1)

    for order in orders:
        try:
            model = ARIMA(prices, order=order)
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_model = fitted
                best_order = order
        except Exception:
            continue

    if best_model is None:
        raise ValueError("Failed to fit any ARIMA model")

    forecast = best_model.forecast(steps=days)
    return forecast.tolist(), list(best_order), best_aic


def main():
    try:
        # 读取 stdin
        input_data = sys.stdin.read()
        data = json.loads(input_data)

        prices = data.get("prices", [])
        days = data.get("days", 5)

        # 检查数据量
        if len(prices) < 30:
            result = {"error": "insufficient data, need >= 30 points"}
            print(json.dumps(result))
            return

        prices = np.array(prices, dtype=np.float64)

        # 优先使用 pmdarima，失败则 fallback 到 statsmodels
        try:
            predictions, order, aic = fit_arima_pmdarima(prices, days)
        except ImportError:
            predictions, order, aic = fit_arima_statsmodels(prices, days)
        except Exception:
            predictions, order, aic = fit_arima_statsmodels(prices, days)

        # 确保预测值合理（非负）
        predictions = [max(0.0, round(p, 2)) for p in predictions]

        result = {
            "predictions": predictions,
            "model_order": order,
            "aic": round(aic, 2),
        }

        print(json.dumps(result))

    except Exception as e:
        result = {"error": str(e)}
        print(json.dumps(result))


if __name__ == "__main__":
    main()
