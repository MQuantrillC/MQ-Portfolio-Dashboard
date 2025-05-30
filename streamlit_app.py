import streamlit as st
import numpy as np
import pandas as pd
from data_loader import get_stock_data

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

# Sidebar
st.sidebar.title("Portfolio Options")
risk = st.sidebar.selectbox("Risk Preference", ["Low", "Medium", "High"])
ticker_input = st.sidebar.text_input("Stock Tickers (comma-separated)", value="AAPL,TSLA,MSFT")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

shares = {}
for ticker in tickers:
    shares[ticker] = st.sidebar.number_input(f"Shares of {ticker}", min_value=0.0, value=1.0, step=1.0, format="%.2f")

apply = st.sidebar.button("Apply")

# Main section
st.title("Portfolio Dashboard")

if apply and tickers:
    data = get_stock_data(tickers, period="6mo")
else:
    data = None

st.subheader("Price Charts")

cols = st.columns([3, 2])

for i, ticker in enumerate(tickers):
    with cols[0]:
        st.markdown(f"#### {ticker} Chart")
        # Placeholder chart: random walk
        chart_data = pd.DataFrame(np.cumsum(np.random.randn(60)), columns=[ticker])
        st.line_chart(chart_data)
    with cols[1]:
        st.markdown(f"**{ticker} Summary**")
        st.write(f"Current Price: $---")
        st.write(f"Close Price: $---")
        st.write(f"Daily Change %: ---")
        st.write(f"Daily Gains/Losses: $---")
        st.write(f"Market Cap: ---")
        st.write(f"Beta: ---")
        st.write(f"PE Ratio: ---")
        st.markdown("---") 