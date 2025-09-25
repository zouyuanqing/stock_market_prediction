from setuptools import setup, find_packages

setup(
    name="stock_market_prediction",
    version="1.0.0",
    author="zouyuanqing",
    author_email="zou.yuanqing@foxmail.com",
    description="A stock market prediction tool based on ARIMA and LSTM models",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zouyuanqing/stock_market_prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "yfinance>=0.2.18",
        "openai==0.28.0",
        "requests>=2.25.0",
        "statsmodels>=0.13.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.15.0",
    ],
    entry_points={
        "console_scripts": [
            "stock-predict=stock_market_prediction.main:main",
        ],
    },
)