"""
Stock Price Prediction — Streamlit Dashboard
Author: Bhavana Suryan
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ──────────────────────────────
st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
    .metric-label { font-size: 0.85rem; color: #8b949e; }
    h1, h2, h3 { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Data Functions ───────────────────────────
@st.cache_data(ttl=3600)
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * std
    df['BB_Lower'] = df['BB_Mid'] - 2 * std
    df['Daily_Return'] = df['Close'].pct_change()
    return df.dropna()


def run_arima(df, steps=30):
    from statsmodels.tsa.arima.model import ARIMA
    close = df['Close']
    model = ARIMA(close, order=(5, 1, 0))
    fitted = model.fit()
    forecast = fitted.forecast(steps=steps)
    ci = fitted.get_forecast(steps=steps).conf_int()

    split = int(len(close) * 0.8)
    rmse = np.sqrt(mean_squared_error(close[split:], fitted.fittedvalues[split:]))
    mae = mean_absolute_error(close[split:], fitted.fittedvalues[split:])
    return fitted.fittedvalues, forecast, ci, rmse, mae


def run_lstm(df, look_back=60, epochs=25):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        close = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close)

        X, y = [], []
        for i in range(look_back, len(scaled)):
            X.append(scaled[i - look_back:i])
            y.append(scaled[i])
        X, y = np.array(X), np.array(y)

        split = int(len(X) * 0.8)
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X[:split], y[:split], epochs=epochs, batch_size=32, verbose=0)

        preds = scaler.inverse_transform(model.predict(X[split:]))
        actual = scaler.inverse_transform(y[split:].reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(actual, preds))
        mae = mean_absolute_error(actual, preds)
        return preds.flatten(), actual.flatten(), df.index[look_back + split:], rmse, mae

    except ImportError:
        return None


# ─── Sidebar ──────────────────────────────────
st.sidebar.title("⚙️ Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
period = st.sidebar.selectbox("Historical Period", ["1y", "2y", "3y", "5y"], index=1)
model_choice = st.sidebar.selectbox("Prediction Model", ["ARIMA", "LSTM", "Both"])
forecast_days = st.sidebar.slider("ARIMA Forecast Days", 10, 90, 30)

st.sidebar.markdown("---")
st.sidebar.markdown("**Popular Tickers**")
for t in ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "RELIANCE.NS", "TCS.NS"]:
    if st.sidebar.button(t):
        ticker = t

# ─── Header ───────────────────────────────────
st.title("📈 StockSense AI — Price Prediction Dashboard")
st.caption(f"Powered by LSTM & ARIMA | Ticker: **{ticker}**")

# ─── Load Data ────────────────────────────────
with st.spinner("Fetching market data..."):
    try:
        df_raw = fetch_data(ticker, period)
        df = add_indicators(df_raw)
        stock_info = yf.Ticker(ticker).info

        name = stock_info.get("longName", ticker)
        st.subheader(f"🏢 {name}")

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

# ─── KPI Metrics ──────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
last = df['Close'].iloc[-1]
prev = df['Close'].iloc[-2]
chg = ((last - prev) / prev) * 100

col1.metric("Current Price", f"${last:.2f}", f"{chg:+.2f}%")
col2.metric("52W High", f"${df['Close'].max():.2f}")
col3.metric("52W Low", f"${df['Close'].min():.2f}")
col4.metric("20-Day SMA", f"${df['SMA_20'].iloc[-1]:.2f}")
col5.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")

# ─── Price Chart with Indicators ──────────────
st.subheader("📊 Price History & Technical Indicators")
fig, axes = plt.subplots(3, 1, figsize=(14, 12), facecolor='#0d1117')
fig.tight_layout(pad=3)

ax1, ax2, ax3 = axes
colors = {'bg': '#0d1117', 'grid': '#21262d', 'price': '#58a6ff',
          'sma20': '#f0883e', 'sma50': '#3fb950', 'bb': '#8b949e'}

for ax in axes:
    ax.set_facecolor(colors['bg'])
    ax.tick_params(colors='#8b949e')
    ax.spines[:].set_color(colors['grid'])
    ax.grid(color=colors['grid'], linestyle='--', alpha=0.5)

# Candlestick / Close Price
ax1.plot(df.index, df['Close'], color=colors['price'], lw=1.5, label='Close')
ax1.plot(df.index, df['SMA_20'], color=colors['sma20'], lw=1.2, label='SMA 20', alpha=0.8)
ax1.plot(df.index, df['SMA_50'], color=colors['sma50'], lw=1.2, label='SMA 50', alpha=0.8)
ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.1, color='#58a6ff', label='Bollinger Bands')
ax1.set_ylabel('Price (USD)', color='#8b949e')
ax1.legend(facecolor='#161b22', labelcolor='#e6edf3', fontsize=8)
ax1.set_title(f'{ticker} — Closing Price', color='#e6edf3', fontsize=12)

# MACD
ax2.plot(df.index, df['MACD'], color='#d2a8ff', lw=1.2)
ax2.axhline(0, color='#8b949e', lw=0.8, linestyle='--')
ax2.fill_between(df.index, df['MACD'], 0, where=df['MACD'] > 0, color='#3fb950', alpha=0.4)
ax2.fill_between(df.index, df['MACD'], 0, where=df['MACD'] < 0, color='#f85149', alpha=0.4)
ax2.set_ylabel('MACD', color='#8b949e')
ax2.set_title('MACD Indicator', color='#e6edf3', fontsize=10)

# RSI
ax3.plot(df.index, df['RSI'], color='#ffa657', lw=1.2)
ax3.axhline(70, color='#f85149', lw=0.8, linestyle='--', label='Overbought (70)')
ax3.axhline(30, color='#3fb950', lw=0.8, linestyle='--', label='Oversold (30)')
ax3.fill_between(df.index, 70, df['RSI'], where=df['RSI'] > 70, color='#f85149', alpha=0.2)
ax3.fill_between(df.index, 30, df['RSI'], where=df['RSI'] < 30, color='#3fb950', alpha=0.2)
ax3.set_ylim(0, 100)
ax3.set_ylabel('RSI', color='#8b949e')
ax3.set_title('RSI (14)', color='#e6edf3', fontsize=10)
ax3.legend(facecolor='#161b22', labelcolor='#e6edf3', fontsize=8)

st.pyplot(fig)

# ─── ARIMA Prediction ─────────────────────────
if model_choice in ["ARIMA", "Both"]:
    st.subheader("🔮 ARIMA Forecast")
    with st.spinner("Fitting ARIMA model..."):
        try:
            fitted_vals, forecast, ci, rmse, mae = run_arima(df, forecast_days)

            fig2, ax = plt.subplots(figsize=(14, 5), facecolor='#0d1117')
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#8b949e')
            ax.spines[:].set_color('#21262d')
            ax.grid(color='#21262d', linestyle='--', alpha=0.5)

            ax.plot(df.index, df['Close'], color='#58a6ff', lw=1.2, label='Actual')
            ax.plot(df.index, fitted_vals, color='#f0883e', lw=1, alpha=0.7, label='ARIMA Fitted')

            future_idx = pd.date_range(df.index[-1], periods=forecast_days + 1, freq='B')[1:]
            ax.plot(future_idx, forecast.values, color='#3fb950', lw=2, label=f'Forecast ({forecast_days}d)')
            ax.fill_between(future_idx, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.2, color='#3fb950')

            ax.set_title(f'ARIMA Forecast — {ticker}', color='#e6edf3')
            ax.legend(facecolor='#161b22', labelcolor='#e6edf3')
            st.pyplot(fig2)

            c1, c2 = st.columns(2)
            c1.metric("ARIMA RMSE", f"{rmse:.4f}")
            c2.metric("ARIMA MAE", f"{mae:.4f}")

        except Exception as e:
            st.warning(f"ARIMA error: {e}")

# ─── LSTM Prediction ──────────────────────────
if model_choice in ["LSTM", "Both"]:
    st.subheader("🧠 LSTM Deep Learning Prediction")
    with st.spinner("Training LSTM model (this may take 1-2 mins)..."):
        result = run_lstm(df)
        if result:
            preds, actual, idx, rmse, mae = result
            fig3, ax = plt.subplots(figsize=(14, 5), facecolor='#0d1117')
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#8b949e')
            ax.spines[:].set_color('#21262d')
            ax.grid(color='#21262d', linestyle='--', alpha=0.5)
            ax.plot(idx, actual, color='#58a6ff', lw=1.5, label='Actual')
            ax.plot(idx, preds, color='#f85149', lw=1.5, label='LSTM Predicted', alpha=0.9)
            ax.set_title(f'LSTM Predictions vs Actual — {ticker}', color='#e6edf3')
            ax.legend(facecolor='#161b22', labelcolor='#e6edf3')
            st.pyplot(fig3)

            c1, c2 = st.columns(2)
            c1.metric("LSTM RMSE", f"{rmse:.4f}")
            c2.metric("LSTM MAE", f"{mae:.4f}")
        else:
            st.warning("Install TensorFlow to enable LSTM: `pip install tensorflow`")

# ─── Correlation Heatmap ──────────────────────
st.subheader("🔗 Feature Correlation Heatmap")
corr_cols = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'MACD', 'RSI', 'Daily_Return']
corr = df[corr_cols].corr()

fig4, ax = plt.subplots(figsize=(9, 6), facecolor='#0d1117')
ax.set_facecolor('#0d1117')
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            linewidths=0.5, linecolor='#21262d',
            annot_kws={'size': 9, 'color': 'white'})
ax.tick_params(colors='#8b949e')
ax.set_title('Feature Correlation Matrix', color='#e6edf3', pad=15)
st.pyplot(fig4)

# ─── Raw Data ─────────────────────────────────
with st.expander("📋 View Raw Data"):
    st.dataframe(df.tail(50).style.background_gradient(cmap='Blues', subset=['Close']))

st.markdown("---")
st.caption("Built by **Bhavana Suryan** | LSTM + ARIMA Hybrid Stock Prediction | B.Tech CSE 2025")
