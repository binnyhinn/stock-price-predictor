import streamlit as st
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

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="centered")

st.title("📈 Stock Price Predictor")
st.markdown("Predict next-day closing prices using **Linear Regression** and technical indicators.")
st.markdown("---")

ticker = st.selectbox("Select Stock", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"])
period = st.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y"])

if st.button("Run Prediction"):
    with st.spinner("Fetching data and training model..."):
        try:
            df = yf.download(ticker, period=period, auto_adjust=True)
            if df.empty:
                st.error("Could not fetch data. Try again.")
                st.stop()

            df["MA5"]  = df["Close"].rolling(5).mean()
            df["MA10"] = df["Close"].rolling(10).mean()
            df["MA20"] = df["Close"].rolling(20).mean()
            df["STD10"] = df["Close"].rolling(10).std()
            df["Return"] = df["Close"].pct_change()
            df["Volume_norm"] = (df["Volume"] - df["Volume"].mean()) / df["Volume"].std()
            df["Target"] = df["Close"].shift(-1)
            df.dropna(inplace=True)

            feature_cols = ["Close", "MA5", "MA10", "MA20", "STD10", "Return", "Volume_norm"]
            X = df[feature_cols].values
            y = df["Target"].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            model = LinearRegression()
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2   = r2_score(y_test, y_pred)

            latest = df[feature_cols].iloc[-1].values.reshape(1, -1)
            next_price = float(model.predict(scaler.transform(latest))[0])
            last_close = float(df["Close"].iloc[-1])
            change     = next_price - last_close
            change_pct = (change / last_close) * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Last Close", f"${last_close:.2f}")
            col2.metric("Next Day Prediction", f"${next_price:.2f}", f"{change_pct:+.2f}%")
            col3.metric("R² Score", f"{r2:.4f}")
            col4.metric("RMSE", f"${rmse:.2f}")

            st.markdown("---")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(y_test,  label="Actual Price",    color="#185FA5", linewidth=1.5)
            ax.plot(y_pred,  label="Predicted Price", color="#3B6D11", linewidth=1.5, linestyle="--")
            ax.axhline(y=next_price, color="#A32D2D", linestyle=":", linewidth=1.2,
                       label=f"Next Day: ${next_price:.2f}")
            ax.set_title(f"{ticker} — Actual vs Predicted Price")
            ax.set_xlabel("Trading Days (Test Set)")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")
