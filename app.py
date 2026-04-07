"""
Stock Price Prediction Web Application
Author: Bhavana Suryan
Tech Stack: Python | Flask | LSTM | ARIMA | TensorFlow/Keras | Streamlit
"""

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch historical stock data using yfinance."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Close', 'Volume', 'Open', 'High', 'Low']]
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, EMA, RSI, Bollinger Bands as features."""
    df = df.copy()

    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Exponential Moving Average
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * std
    df['BB_Lower'] = df['BB_Mid'] - 2 * std

    # Daily Return
    df['Daily_Return'] = df['Close'].pct_change()

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────
# LSTM MODEL
# ─────────────────────────────────────────────
def build_lstm_model(input_shape):
    """Build and compile LSTM model."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        return model
    except ImportError:
        return None


def prepare_lstm_data(series: np.ndarray, look_back: int = 60):
    """Create sliding window sequences for LSTM."""
    X, y = [], []
    for i in range(look_back, len(series)):
        X.append(series[i - look_back:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def run_lstm_prediction(df: pd.DataFrame, look_back: int = 60, epochs: int = 30):
    """Train LSTM and return predictions + metrics."""
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)

    X, y = prepare_lstm_data(scaled, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model((look_back, 1))
    if model is None:
        return None

    model.fit(X_train, y_train, epochs=epochs, batch_size=32,
              validation_split=0.1, verbose=0)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions))
    mae = mean_absolute_error(y_test_inv, predictions)

    return {
        "predictions": predictions.flatten().tolist(),
        "actual": y_test_inv.flatten().tolist(),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "model": "LSTM",
        "index": df.index[look_back + split:].strftime('%Y-%m-%d').tolist()
    }


# ─────────────────────────────────────────────
# ARIMA MODEL
# ─────────────────────────────────────────────
def run_arima_prediction(df: pd.DataFrame, steps: int = 30):
    """Fit ARIMA and forecast future prices."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller

        close = df['Close']

        # ADF Test for stationarity
        adf_result = adfuller(close)
        d = 1 if adf_result[1] > 0.05 else 0

        # Fit ARIMA(5, d, 0)
        model = ARIMA(close, order=(5, d, 0))
        fitted = model.fit()

        # In-sample fitted
        fitted_values = fitted.fittedvalues

        # Forecast
        forecast = fitted.forecast(steps=steps)
        conf_int = fitted.get_forecast(steps=steps).conf_int()

        split = int(len(close) * 0.8)
        test_actual = close[split:]
        test_fitted = fitted_values[split:]

        rmse = np.sqrt(mean_squared_error(test_actual, test_fitted))
        mae = mean_absolute_error(test_actual, test_fitted)

        return {
            "forecast": forecast.tolist(),
            "forecast_lower": conf_int.iloc[:, 0].tolist(),
            "forecast_upper": conf_int.iloc[:, 1].tolist(),
            "fitted": fitted_values.tolist(),
            "actual": close.tolist(),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "model": "ARIMA",
            "index": close.index.strftime('%Y-%m-%d').tolist()
        }
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get('ticker', 'AAPL').upper()
    model_type = data.get('model', 'arima')

    try:
        df = fetch_stock_data(ticker)
        df = add_technical_indicators(df)

        if model_type == 'lstm':
            result = run_lstm_prediction(df)
        else:
            result = run_arima_prediction(df)

        if result is None:
            return jsonify({"error": "TensorFlow not installed. Use ARIMA mode."}), 400

        result['ticker'] = ticker
        result['last_price'] = round(df['Close'].iloc[-1], 2)
        result['price_change'] = round(df['Daily_Return'].iloc[-1] * 100, 2)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/stock_info', methods=['GET'])
def stock_info():
    ticker = request.args.get('ticker', 'AAPL').upper()
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return jsonify({
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
