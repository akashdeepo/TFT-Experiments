# Temporal Fusion Transformer (TFT) Backtesting Framework

## Overview
This repository contains an experimental implementation of a **Temporal Fusion Transformer (TFT)** framework for financial time series forecasting and backtesting. It is designed to integrate state-of-the-art modeling techniques with a robust backtesting engine to evaluate trading strategies based on model predictions.

The project includes:
- **Data preprocessing** for financial time series.
- **TFT model implementation** for forecasting multiple quantiles.
- **Trading strategy** leveraging model outputs for decision-making.
- **Backtesting engine** for performance evaluation and risk analysis.

This framework is suitable for researchers, practitioners, and enthusiasts looking to experiment with advanced forecasting models and evaluate their trading performance on historical data.

---

## Key Features
- **Feature Engineering**: Automated creation of technical indicators like returns, volatility, and moving averages.
- **Temporal Fusion Transformer**: Custom implementation of the TFT model, including:
  - Variable Selection Network.
  - Gated Residual Network.
  - Multi-head Attention Mechanism.
  - Quantile Regression.
- **Trading Strategy**:
  - Dynamic risk management.
  - Signal generation based on model predictions.
  - Stop-loss and take-profit mechanisms.
  - Enhanced filters for strong signals.
- **Backtesting Engine**:
  - Supports rolling-window predictions.
  - Tracks PnL, win rate, Sharpe ratio, and max drawdown.
  - Allows detailed trade analytics and visualization.

---

## Requirements
### Dependencies:
- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - tqdm
  - torch

Install required packages via:
```bash
pip install -r requirements.txt
```

### Hardware:
- A CUDA-enabled GPU is recommended for training the TFT model.
- Sufficient memory for handling large datasets.

---


## Usage
### 1. Data Preparation
Prepare a CSV file containing minute-level OHLCV data and store it in the `data/` directory. The required columns are:
- `timestamp` (datetime index)
- `open`
- `high`
- `low`
- `close`
- `volume`

Run the `preprocessing.py` script to generate engineered features:
```bash
python src/preprocessing.py
```

### 2. Model Training
Train the Temporal Fusion Transformer model on historical data:
```bash
python src/model.py
```

### 3. Backtesting
Run the backtesting engine with the trained model:
```bash
python main.py
```

### 4. Results Analysis
Analyze the backtest results using the provided visualization functions. This includes:
- Cumulative PnL
- Trade distribution
- Feature importance
- Attention patterns

---



## Acknowledgments
- [PyTorch](https://pytorch.org/)
- [Temporal Fusion Transformer Paper](https://arxiv.org/abs/1912.09363)
- Financial datasets used for experimental analysis.

---

Happy forecasting and backtesting!

