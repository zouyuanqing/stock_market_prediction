from setuptools import setup, find_packages

setup(
    name="stock_market_prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "yfinance",
        "openai",
        "requests",
        "statsmodels",
        "scikit-learn",
        "tensorflow",
    ],
    entry_points={
        "console_scripts": [
            "stock_market_prediction=stock_market_prediction.main:main",
        ],
    },
)
