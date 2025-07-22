import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Manual EMA function
def manual_ema(series, length):
    if len(series) < length or series.isna().all():
        return pd.Series(np.nan, index=series.index)
    alpha = 2 / (length + 1)
    ema = series.copy()
    ema.iloc[:length-1] = np.nan  # Initialize with NaN for initial period
    ema.iloc[length-1] = series.iloc[:length].mean()  # First EMA as mean
    for i in range(length, len(series)):
        if np.isnan(ema.iloc[i-1]) or np.isnan(series.iloc[i]):
            ema.iloc[i] = np.nan
        else:
            ema.iloc[i] = (series.iloc[i] * alpha) + (ema.iloc[i-1] * (1 - alpha))
    return ema

# Manual RSI function
def manual_rsi(series, length):
    if len(series) < length + 1 or series.isna().all():
        return pd.Series(np.nan, index=series.index)
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=length, min_periods=length).mean()
    avg_loss = loss.rolling(window=length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Manual pivot points function
def manual_pivotpoints(high, low, close, open_):
    if len(high) == 0 or high.isna().all() or low.isna().all() or close.isna().all():
        return pd.DataFrame({'S1': pd.Series([], dtype=float), 'R1': pd.Series([], dtype=float)})
    pp = (high + low + close) / 3
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    return pd.DataFrame({'S1': s1, 'R1': r1}, index=high.index)

# Function to compute QQE components
def compute_qqe_components(df, rsi_period=6, sf=5, qqe_factor=3):
    if df.empty or 'Close' not in df.columns or len(df) < max(rsi_period, sf, qqe_factor) or df['Close'].isna().all():
        return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
    close = df['Close']
    rsi_val = manual_rsi(close, rsi_period)
    if rsi_val.isna().all():
        return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
    rsi_ma = manual_ema(rsi_val, sf)
    atr_rsi = np.abs(rsi_ma.diff())
    wilders_period = rsi_period * 2 - 1
    ma_atr_rsi = manual_ema(atr_rsi, wilders_period)
    dar = manual_ema(ma_atr_rsi, wilders_period) * qqe_factor if not ma_atr_rsi.isna().all() else pd.Series(np.nan, index=df.index)
    longband = pd.Series(np.nan, index=df.index)
    shortband = pd.Series(np.nan, index=df.index)
    trend = pd.Series(np.nan, index=df.index)
    trend.iloc[0] = 1.0
    if not np.isnan(rsi_ma.iloc[0]) and not np.isnan(dar.iloc[0]):
        longband.iloc[0] = rsi_ma.iloc[0] - dar.iloc[0]
        shortband.iloc[0] = rsi_ma.iloc[0] + dar.iloc[0]
    for i in range(1, len(df)):
        new_longband = rsi_ma.iloc[i] - dar.iloc[i] if not np.isnan(rsi_ma.iloc[i]) and not np.isnan(dar.iloc[i]) else np.nan
        new_shortband = rsi_ma.iloc[i] + dar.iloc[i] if not np.isnan(rsi_ma.iloc[i]) and not np.isnan(dar.iloc[i]) else np.nan
        if not np.isnan(rsi_ma.iloc[i-1]) and not np.isnan(longband.iloc[i-1]) and rsi_ma.iloc[i-1] > longband.iloc[i-1] and rsi_ma.iloc[i] > longband.iloc[i-1]:
            longband.iloc[i] = max(longband.iloc[i-1], new_longband)
        else:
            longband.iloc[i] = new_longband
        if not np.isnan(rsi_ma.iloc[i-1]) and not np.isnan(shortband.iloc[i-1]) and rsi_ma.iloc[i-1] < shortband.iloc[i-1] and rsi_ma.iloc[i] < shortband.iloc[i-1]:
            shortband.iloc[i] = min(shortband.iloc[i-1], new_shortband)
        else:
            shortband.iloc[i] = new_shortband
        cross_short = (rsi_ma.iloc[i-1] - shortband.iloc[i-1]) * (rsi_ma.iloc[i] - shortband.iloc[i]) < 0 if not np.isnan(shortband.iloc[i-1]) else False
        cross_long = (longband.iloc[i-1] - rsi_ma.iloc[i-1]) * (longband.iloc[i-1] - rsi_ma.iloc[i]) < 0 if not np.isnan(longband.iloc[i-1]) else False
        trend.iloc[i] = 1 if cross_short else -1 if cross_long else trend.iloc[i-1]
    fast_atr_rsi_tl = np.where(trend == 1, longband, shortband)
    return rsi_ma, fast_atr_rsi_tl, trend

# Function to get data and resample if needed
def get_data(symbol, interval, period):
    try:
        raw_df = yf.download(symbol, interval='1h' if interval == '4h' else interval, period=period, auto_adjust=True)
        if raw_df.empty:
            return raw_df
        if interval == '4h':
            raw_df = raw_df.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        return raw_df.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# App layout
st.title("Trading Analysis App")

# Asset selection
assets = {
    'Forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X'],
    'Indices': ['^GSPC', '^DJI', '^IXIC', '^FTSE'],
    'Commodities': ['GC=F', 'SI=F', 'CL=F', 'NG=F'],
    'Crypto': ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD']
}
category = st.selectbox("Select Category", list(assets.keys()))
symbol = st.selectbox("Select Symbol", assets[category])

# Timeframe for chart
timeframes = ['Hourly', '4 Hourly', 'Daily', 'Weekly']
selected_tf = st.selectbox("Select Timeframe for Chart", timeframes)
interval_map = {'Hourly': '1h', '4 Hourly': '4h', 'Daily': '1d', 'Weekly': '1wk'}
period_map = {'1h': '60d', '4h': '60d', '1d': '5y', '1wk': '10y'}
df = get_data(symbol, interval_map[selected_tf], period_map[interval_map[selected_tf]])

if df.empty:
    st.error("No data available for this symbol and timeframe.")
else:
    # Compute indicators
    df['EMA8'] = manual_ema(df['Close'], 8)
    df['EMA20'] = manual_ema(df['Close'], 20)
    df['EWO'] = manual_ema(df['Close'], 5) - manual_ema(df['Close'], 35)

    rsi_ma1, fast_tl1, trend1 = compute_qqe_components(df, 6, 5, 3)
    rsi_ma2, fast_tl2, trend2 = compute_qqe_components(df, 6, 5, 1.61)

    if rsi_ma1.isna().all() or fast_tl1.isna().all() or trend1.isna().all() or rsi_ma2.isna().all() or fast_tl2.isna().all() or trend2.isna().all():
        st.error("Insufficient data for QQE calculation.")
    else:
        length = 50
        mult = 0.35
        basis = (fast_tl1 - 50).rolling(window=length, min_periods=1).mean()
        dev = mult * (fast_tl1 - 50).rolling(window=length, min_periods=1).std()
        upper = basis + dev
        lower = basis - dev

        thresh2 = 3
        histo = rsi_ma2 - 50
        hcolor = np.where(histo > thresh2, 'silver', np.where(histo < -thresh2, 'silver', 'rgba(0,0,0,0)'))
        line = fast_tl2 - 50

        greenbar1 = histo > thresh2
        greenbar2 = (rsi_ma1 - 50) > upper
        green_bar = np.where(greenbar1 & greenbar2, histo, np.nan)

        redbar1 = histo < -thresh2
        redbar2 = (rsi_ma1 - 50) < lower
        red_bar = np.where(redbar1 & redbar2, histo, np.nan)

        # Multi-timeframe analysis
        st.subheader("Trends and Support/Resistance Levels")
        tf_intervals = {'Weekly': '1wk', 'Daily': '1d', '4 Hourly': '4h', 'Hourly': '1h'}
        data = []
        for tf, intv in tf_intervals.items():
            df_tf = get_data(symbol, intv, period_map.get(intv, '10y')).tail(200)
            if not df_tf.empty:
                _, _, trend_tf = compute_qqe_components(df_tf, 6, 5, 1.61)
                trend_str = "Bullish" if trend_tf.iloc[-1] == 1 else "Bearish" if not np.isnan(trend_tf.iloc[-1]) else "N/A"
                pp = manual_pivotpoints(df_tf['High'], df_tf['Low'], df_tf['Close'], df_tf['Open'])
                support = pp['S1'].iloc[-1] if not np.isnan(pp['S1'].iloc[-1]) else "N/A"
                resistance = pp['R1'].iloc[-1] if not np.isnan(pp['R1'].iloc[-1]) else "N/A"
                data.append([tf, trend_str, support, resistance])
        st.table(pd.DataFrame(data, columns=["Timeframe", "Trend", "Support", "Resistance"]))

        # Price action
        st.subheader("Price Action")
        last_candle_bullish = "Bullish" if df['Close'].iloc[-1] > df['Open'].iloc[-1] else "Bearish"
        st.write(f"Last Candle Trend: {last_candle_bullish}")
        patterns = []
        if len(df) >= 2:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            if last['Close'] > prev['Open'] and last['Open'] < prev['Close'] and last['Close'] > prev['High'] and last['Open'] < prev['Low']:
                patterns.append("Bullish Engulfing")
            elif last['Close'] < prev['Open'] and last['Open'] > prev['Close'] and last['Close'] < prev['Low'] and last['Open'] > prev['High']:
                patterns.append("Bearish Engulfing")
        if patterns:
            st.write("Detected Candle Patterns: " + ", ".join(patterns))
        else:
            st.write("No specific candle patterns detected in the last bar.")

        # Chart
        st.subheader("Active Chart")
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=("Price Chart", "QQE MOD", "Elliott Wave Oscillator"))

        # Price chart
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA8'], name="EMA8", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name="EMA20", line=dict(color="blue")), row=1, col=1)

        # QQE MOD
        fig.add_trace(go.Bar(x=df.index, y=histo, name="Histo", marker=dict(color=hcolor)), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=green_bar, name="QQE Up", marker_color="#00c3ff"), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=red_bar, name="QQE Down", marker_color="#ff0062"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=line, name="QQE Line", line=dict(color="white")), row=2, col=1)
        fig.add_hline(y=0, row=2, col=1, line_dash="dot", line_color="white")

        # Elliott Wave Oscillator
        ewo_colors = np.where(df['EWO'].notna(), np.where(df['EWO'] > 0, 'green', 'red'), 'gray')
        fig.add_trace(go.Bar(x=df.index, y=df['EWO'], name="EWO", marker=dict(color=ewo_colors)), row=3, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="QQE", row=2, col=1)
        fig.update_yaxes(title_text="EWO", row=3, col=1)
        st.plotly_chart(fig)

        # Suggestions
        st.subheader("Suggested Entry, Take Profit, Stop Loss (Based on Selected TF)")
        overall_trend = "Bullish" if trend2.iloc[-1] == 1 else "Bearish" if not np.isnan(trend2.iloc[-1]) else "N/A"
        current_price = df['Close'].iloc[-1]
        pp_main = manual_pivotpoints(df['High'], df['Low'], df['Close'], df['Open'])
        support_main = pp_main['S1'].iloc[-1] if not np.isnan(pp_main['S1'].iloc[-1]) else current_price * 0.99
        resistance_main = pp_main['R1'].iloc[-1] if not np.isnan(pp_main['R1'].iloc[-1]) else current_price * 1.01
        if overall_trend == "Bullish":
            entry = current_price
            tp = resistance_main
            sl = support_main
        else:
            entry = current_price
            tp = support_main
            sl = resistance_main
        st.write(f"Trend: {overall_trend}")
        st.write(f"Suggested Entry: {entry:.5f}")
        st.write(f"Suggested Take Profit: {tp:.5f}")
        st.write(f"Suggested Stop Loss: {sl:.5f}")
