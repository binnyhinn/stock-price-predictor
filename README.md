# Stock Price Predictor

A machine learning project that predicts next-day stock closing prices using **Linear Regression** and technical indicators.

Built with Python, scikit-learn, and yfinance.

---

## Features

- Fetches real-time historical stock data via `yfinance`
- Engineers technical features: Moving Averages (MA5, MA10, MA20), standard deviation, daily returns, volume
- Trains a Linear Regression model with an 80/20 train-test split
- Evaluates model performance using R² Score and RMSE
- Predicts next-day closing price
- Plots actual vs predicted prices and saves the chart

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| scikit-learn | Linear Regression, preprocessing, evaluation |
| yfinance | Fetch real stock data from Yahoo Finance |
| pandas / numpy | Data manipulation and feature engineering |
| matplotlib | Visualization |

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/stock-price-predictor.git
cd stock-price-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the predictor
```bash
python stock_predictor.py
```

You'll be prompted to enter a stock ticker and time period:
```
Enter stock ticker (e.g. AAPL, TSLA, MSFT): AAPL
Data period (1mo / 3mo / 6mo / 1y) [default: 6mo]:
```

---

## Sample Output

```
Fetched 126 rows of data.

Model Performance:
  R² Score : 0.9821
  RMSE     : $2.34

Prediction for AAPL:
  Last Close Price  : $189.30
  Next Day Predicted: $191.45
  Expected Change   : +$2.15 (+1.14%)

Plot saved as prediction_plot.png
```

---

## Project Structure

```
stock-price-predictor/
├── stock_predictor.py   # Main script
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## What I Learned

- Feature engineering with financial time-series data
- Supervised ML pipeline: data → features → train → evaluate → predict
- Scikit-learn's LinearRegression and StandardScaler
- Model evaluation metrics: R² and RMSE
- Working with real-world financial data APIs

---

## Disclaimer

This project is for **educational purposes only**. It is not financial advice and should not be used for actual trading decisions.

---

## Author

Built by [Binnyhinn] — AI/ML enthusiast

