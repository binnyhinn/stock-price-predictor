import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def fetch_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch historical stock data using yfinance."""
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    print(f"Fetched {len(df)} rows of data.")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator features for the model."""
    df = df.copy()
    df["MA5"]  = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["STD10"] = df["Close"].rolling(window=10).std()
    df["Return"] = df["Close"].pct_change()
    df["Volume_norm"] = (df["Volume"] - df["Volume"].mean()) / df["Volume"].std()
    df["Target"] = df["Close"].shift(-1)  # next day's closing price
    df.dropna(inplace=True)
    return df


def train_model(df: pd.DataFrame):
    """Train a Linear Regression model and return results."""
    feature_cols = ["Close", "MA5", "MA10", "MA20", "STD10", "Return", "Volume_norm"]
    X = df[feature_cols].values
    y = df["Target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2    = r2_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"  R² Score : {r2:.4f}")
    print(f"  RMSE     : ${rmse:.2f}")

    return model, scaler, X_test_scaled, y_test, y_pred, feature_cols


def predict_next_day(model, scaler, df: pd.DataFrame, feature_cols: list) -> float:
    """Predict the next day's closing price."""
    latest = df[feature_cols].iloc[-1].values.reshape(1, -1)
    latest_scaled = scaler.transform(latest)
    prediction = model.predict(latest_scaled)[0]
    return prediction


def plot_results(ticker: str, y_test: np.ndarray, y_pred: np.ndarray, next_price: float):
    """Plot actual vs predicted prices."""
    plt.figure(figsize=(12, 5))
    plt.plot(y_test,  label="Actual Price",    color="#185FA5", linewidth=1.5)
    plt.plot(y_pred,  label="Predicted Price", color="#3B6D11", linewidth=1.5, linestyle="--")
    plt.axhline(y=next_price, color="#A32D2D", linestyle=":", linewidth=1.2,
                label=f"Next Day Prediction: ${next_price:.2f}")
    plt.title(f"{ticker} — Stock Price Prediction (Linear Regression)")
    plt.xlabel("Trading Days (Test Set)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_plot.png", dpi=150)
    plt.show()
    print("\nPlot saved as prediction_plot.png")


def main():
    ticker = input("Enter stock ticker (e.g. AAPL, TSLA, MSFT): ").upper().strip()
    period = input("Data period (1mo / 3mo / 6mo / 1y) [default: 6mo]: ").strip() or "6mo"

    df = fetch_data(ticker, period)
    df = add_features(df)

    model, scaler, X_test, y_test, y_pred, feature_cols = train_model(df)

    next_price = predict_next_day(model, scaler, df, feature_cols)
    last_close = float(df["Close"].iloc[-1])
    change     = next_price - last_close
    change_pct = (change / last_close) * 100

    print(f"\nPrediction for {ticker}:")
    print(f"  Last Close Price  : ${last_close:.2f}")
    print(f"  Next Day Predicted: ${next_price:.2f}")
    print(f"  Expected Change   : ${change:+.2f} ({change_pct:+.2f}%)")

    plot_results(ticker, y_test, y_pred, next_price)


if __name__ == "__main__":
    main()
