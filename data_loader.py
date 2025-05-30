import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data(tickers, period="1y"):
    """
    Fetch historical stock data for given tickers.
    Always returns a DataFrame with MultiIndex columns: (Field, Ticker)
    """
    try:
        data = yf.download(tickers, period=period, group_by='ticker')
        # If only one ticker, convert columns to MultiIndex
        if isinstance(tickers, str) or (isinstance(tickers, list) and len(tickers) == 1):
            ticker = tickers if isinstance(tickers, str) else tickers[0]
            data.columns = pd.MultiIndex.from_product([data.columns, [ticker]])
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def get_current_prices(tickers):
    """
    Get current prices for given tickers
    """
    try:
        data = yf.download(tickers, period="1d", interval="1m")
        return data['Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching current prices: {e}")
        return None

def calculate_portfolio_value(holdings):
    """
    Calculate current portfolio value based on holdings
    holdings: dict with ticker as key and number of shares as value
    """
    if not holdings:
        return 0
    
    current_prices = get_current_prices(list(holdings.keys()))
    if current_prices is None:
        return 0
    
    total_value = 0
    for ticker, shares in holdings.items():
        if ticker in current_prices:
            total_value += shares * current_prices[ticker]
    
    return total_value 