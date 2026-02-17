import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

st.set_page_config("🚀 AI Trading Algorithms Lab", layout="wide")

# =====================================================
# SIDEBAR - STOCK FILTERS
# =====================================================
st.sidebar.title("🔎 Stock Filters")

symbol = st.sidebar.text_input("Stock Symbol", "RELIANCE.NS")
min_volume = st.sidebar.slider("Minimum Volume", 0, 5000000, 100000)
min_rsi = st.sidebar.slider("Minimum RSI", 0, 100, 40)

# =====================================================
# SIDEBAR - ALGORITHM FILTER
# =====================================================
st.sidebar.title("🧠 Algorithm Selection")

algo_category = st.sidebar.radio(
    "Algorithm Type",
    ["Rule-Based", "Machine Learning", "All"]
)

rule_algos = [
    "RSI Mean Reversion",
    "MACD Crossover",
    "SMA Trend Following",
    "Bollinger Breakout",
    "Momentum + Volume"
]

ml_algos = [
    "Linear Regression",
    "Random Forest",
    "Extra Trees",
    "KNN",
    "XGBoost"
]

if algo_category == "Rule-Based":
    selected_algos = st.sidebar.multiselect(
        "Select Rule-Based Strategies",
        rule_algos,
        default=["RSI Mean Reversion"]
    )

elif algo_category == "Machine Learning":
    selected_algos = st.sidebar.multiselect(
        "Select ML Models",
        ml_algos,
        default=["Random Forest"]
    )

else:
    selected_algos = st.sidebar.multiselect(
        "Select Algorithms",
        rule_algos + ml_algos,
        default=["RSI Mean Reversion"]
    )

strategy_mode = st.sidebar.selectbox(
    "Execution Mode",
    ["Single Strategy", "Ensemble Voting"]
)

# =====================================================
# FETCH DATA (1D SAFE)
# =====================================================
@st.cache_data
def fetch(symbol):
    df = yf.download(symbol, period="30d", interval="15m", progress=False)

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = pd.Series(df[col].values.ravel())

    return df.dropna()

df = fetch(symbol)

if df is None:
    st.error("No data found")
    st.stop()

# =====================================================
# INDICATORS
# =====================================================
close = df["Close"]

df["RSI"] = RSIIndicator(close, 14).rsi()

macd = MACD(close)
df["MACD"] = macd.macd()
df["MACD_SIGNAL"] = macd.macd_signal()

df["SMA20"] = SMAIndicator(close, 20).sma_indicator()
df["SMA50"] = SMAIndicator(close, 50).sma_indicator()

bb = BollingerBands(close)
df["BB_HIGH"] = bb.bollinger_hband()
df["BB_LOW"] = bb.bollinger_lband()

df = df.dropna()

# Apply filters
df = df[df["Volume"] > min_volume]
df = df[df["RSI"] > min_rsi]

# =====================================================
# SIGNAL ENGINE
# =====================================================
df["SignalScore"] = 0
confidence = None

for algo in selected_algos:

    # RULE BASED
    if algo == "RSI Mean Reversion":
        df.loc[df["RSI"] < 30, "SignalScore"] += 1
        df.loc[df["RSI"] > 70, "SignalScore"] -= 1

    elif algo == "MACD Crossover":
        df.loc[df["MACD"] > df["MACD_SIGNAL"], "SignalScore"] += 1
        df.loc[df["MACD"] < df["MACD_SIGNAL"], "SignalScore"] -= 1

    elif algo == "SMA Trend Following":
        df.loc[df["SMA20"] > df["SMA50"], "SignalScore"] += 1
        df.loc[df["SMA20"] < df["SMA50"], "SignalScore"] -= 1

    elif algo == "Bollinger Breakout":
        df.loc[df["Close"] < df["BB_LOW"], "SignalScore"] += 1
        df.loc[df["Close"] > df["BB_HIGH"], "SignalScore"] -= 1

    elif algo == "Momentum + Volume":
        df.loc[
            (df["RSI"] > 55) &
            (df["Volume"] > df["Volume"].rolling(20).mean()),
            "SignalScore"
        ] += 1

    # ML MODELS
    elif algo in ml_algos:

        df_ml = df.copy()
        df_ml["Target"] = df_ml["Close"].shift(-1)
        df_ml = df_ml.dropna()

        features = ["Close","Volume","RSI","MACD"]
        X = df_ml[features]
        y = df_ml["Target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,y,test_size=0.2,shuffle=False
        )

        if algo == "Linear Regression":
            model = LinearRegression()
        elif algo == "Random Forest":
            model = RandomForestRegressor()
        elif algo == "Extra Trees":
            model = ExtraTreesRegressor()
        elif algo == "KNN":
            model = KNeighborsRegressor(n_neighbors=min(5,len(X_train)))
        elif algo == "XGBoost":
            model = XGBRegressor(objective="reg:squarederror")

        model.fit(X_train,y_train)

        pred = model.predict(X.iloc[-1:].values)[0]
        confidence = round(r2_score(y_test, model.predict(X_test))*100,2)

        if pred > df["Close"].iloc[-1]:
            df.loc[df.index[-1], "SignalScore"] += 1
        else:
            df.loc[df.index[-1], "SignalScore"] -= 1

# =====================================================
# FINAL DECISION
# =====================================================
latest_score = df["SignalScore"].iloc[-1]

if strategy_mode == "Single Strategy":
    final_signal = "BUY" if latest_score > 0 else "SELL"
else:
    if latest_score > 0:
        final_signal = "BUY"
    elif latest_score < 0:
        final_signal = "SELL"
    else:
        final_signal = "HOLD"

# =====================================================
# VISUALIZATION
# =====================================================
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    row_heights=[0.6,0.2,0.2]
)

fig.add_trace(go.Scatter(x=df["Datetime"], y=df["Close"], name="Price"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["Datetime"], y=df["SMA20"], name="SMA20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["Datetime"], y=df["SMA50"], name="SMA50"), row=1, col=1)

fig.add_trace(go.Scatter(x=df["Datetime"], y=df["RSI"], name="RSI"), row=2, col=1)

fig.add_trace(go.Scatter(x=df["Datetime"], y=df["MACD"], name="MACD"), row=3, col=1)
fig.add_trace(go.Scatter(x=df["Datetime"], y=df["MACD_SIGNAL"], name="MACD Signal"), row=3, col=1)

fig.update_layout(height=900, title=f"📊 AI Trading Dashboard - {symbol}")

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# OUTPUT PANEL
# =====================================================
current_price = df["Close"].iloc[-1]
atm_strike = round(current_price/50)*50

col1,col2,col3 = st.columns(3)

col1.metric("Current Price", round(current_price,2))
col2.metric("Final Signal", final_signal)
col3.metric("ATM Option", f"{atm_strike} {'CE' if final_signal=='BUY' else 'PE'}")

if confidence:
    st.metric("🧠 ML Confidence %", confidence)

st.caption("⚡ Multi-Algorithm Engine | Rule-Based + ML | Production Ready")
