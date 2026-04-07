# 📈 StockSense AI — Stock Price Prediction using LSTM & ARIMA

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

> A production-grade hybrid ML application combining **LSTM (Deep Learning)** and **ARIMA (Statistical)** models for stock price forecasting, with both a Flask REST API backend and an interactive Streamlit dashboard.

---

## 🚀 Live Demo

| Interface | Command |
|-----------|---------|
| Streamlit Dashboard | `streamlit run streamlit_app.py` |
| Flask Web App | `python app.py` → visit `http://localhost:5000` |

---

## 🎯 Features

- **Dual Model Architecture** — LSTM neural network + ARIMA statistical model
- **Technical Indicators** — SMA (20/50), EMA (12/26), MACD, RSI (14), Bollinger Bands
- **Model Evaluation** — RMSE and MAE metrics on test split
- **ARIMA Forecasting** — Confidence intervals for future price ranges
- **Interactive Dashboard** — Real-time charts with Streamlit
- **REST API** — Flask endpoints for programmatic access
- **Real-time Data** — Live market data via `yfinance`
- **Feature Correlation Heatmap** — Seaborn-based visual analysis

---

## 🏗️ Architecture

```
stock-prediction/
├── app.py                  # Flask REST API backend
├── streamlit_app.py        # Interactive Streamlit dashboard
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Flask frontend
└── README.md
```

### Model Pipeline

```
Raw Data (yfinance)
      │
      ▼
Feature Engineering
(SMA, EMA, MACD, RSI, Bollinger Bands)
      │
      ├──────────────┬──────────────────
      ▼              ▼
   ARIMA           LSTM
  (p=5, d=1)    (128→64→32 units)
      │              │
      ▼              ▼
  Forecast       Test Predictions
      │              │
      └──────────────┘
             │
             ▼
       RMSE / MAE
       Evaluation
```

---

## 🧠 Models Explained

### LSTM (Long Short-Term Memory)
- 3-layer LSTM with `BatchNormalization` and `Dropout (0.2)` for regularization
- Uses **60-day look-back window** as sequence input
- Optimized with **Huber Loss** (robust to outliers)
- Train/Test split: **80/20**

### ARIMA (AutoRegressive Integrated Moving Average)
- Order: `(5, 1, 0)` — ADF test determines differencing
- Generates **confidence intervals** for forecast range
- Fits in-sample values for residual analysis

---

## 📊 Sample Metrics (AAPL, 2yr)

| Model | RMSE | MAE |
|-------|------|-----|
| LSTM | ~4.2 | ~3.1 |
| ARIMA | ~5.8 | ~4.3 |

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Bhavanasuryan/stock-price-prediction.git
cd stock-price-prediction
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```

### 5. Run Flask App (API + Web UI)
```bash
python app.py
# Visit: http://localhost:5000
```

---

## 🔌 API Reference (Flask)

### POST `/predict`
```json
{
  "ticker": "AAPL",
  "model": "arima"    // or "lstm"
}
```

**Response:**
```json
{
  "model": "ARIMA",
  "ticker": "AAPL",
  "last_price": 189.34,
  "price_change": 0.82,
  "rmse": 5.2143,
  "mae": 4.0021,
  "forecast": [...],
  "actual": [...],
  "index": [...]
}
```

### GET `/stock_info?ticker=AAPL`
Returns sector, market cap, P/E ratio, 52-week high/low.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow 2.x, Keras (LSTM) |
| Statistical ML | statsmodels (ARIMA) |
| Data | yfinance, pandas, numpy |
| Visualization | matplotlib, seaborn |
| Backend | Flask 3.0 |
| Dashboard | Streamlit 1.36 |
| Evaluation | scikit-learn (RMSE, MAE) |

---

## 📈 Technical Indicators Used

| Indicator | Purpose |
|-----------|---------|
| SMA 20/50 | Trend direction |
| EMA 12/26 | Momentum-weighted average |
| MACD | Trend momentum signal |
| RSI (14) | Overbought/oversold detection |
| Bollinger Bands | Volatility range |
| Daily Return | Short-term movement |

---

## 🔮 Future Improvements

- [ ] Add Transformer / Attention mechanism
- [ ] Multi-stock portfolio comparison
- [ ] Sentiment analysis (news + Reddit)
- [ ] Docker containerization
- [ ] Deploy on AWS/GCP with CI/CD pipeline

---

## 👤 Author

**Bhavana Suryan**  
B.Tech Computer Science | BKIT, Karnataka | 2025  
📧 suryanbhavana691@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/bhavanasuryan) | [GitHub](https://github.com/Bhavanasuryan)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

⭐ **If this project helped you, please give it a star!**
