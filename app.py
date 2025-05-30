import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime
import requests
import os
import plotly.express as px
from collections import Counter
import traceback
import concurrent.futures
from functools import lru_cache
from numbers import Number
from scipy.optimize import minimize
from typing import Tuple

# Initialize all required session state variables
required_state_vars = [
    'financial_data_cache', 'portfolio_created', 'show_charts',
    'show_financials', 'show_monte_carlo', 'show_market_overview',
    'custom_weight_inputs', 'ticker_info_cache', 'optimal_result',
    'optimal_metrics', 'optimal_backtest', 'optimal_table',
    'sector_map', 'missing_tickers', 'selected_tickers',
    'risk_level', 'close_prices', 'portfolio_weights', 'ef_portfolios'
]

for var in required_state_vars:
    if var not in st.session_state:
        if var in ['financial_data_cache', 'ticker_info_cache']:
            st.session_state[var] = {}
        elif var in ['portfolio_created', 'show_charts', 'show_financials', 
                    'show_monte_carlo', 'show_market_overview']:
            st.session_state[var] = False
        else:
            st.session_state[var] = None

@st.cache_data(ttl=3600)
def get_historical_closes(ticker):
    """Get historical close prices for different time periods"""
    try:
        # Get current price
        current_price = yf.Ticker(to_yahoo_ticker(ticker)).history(period="1d")["Close"].iloc[-1]
        
        # Get historical prices
        five_d_ago = yf.Ticker(to_yahoo_ticker(ticker)).history(period="5d")["Close"].iloc[0]
        one_m_ago = yf.Ticker(to_yahoo_ticker(ticker)).history(period="1mo")["Close"].iloc[0]
        six_m_ago = yf.Ticker(to_yahoo_ticker(ticker)).history(period="6mo")["Close"].iloc[0]
        
        return {
            'current': current_price,
            '5d': five_d_ago,
            '1m': one_m_ago,
            '6m': six_m_ago
        }
    except Exception as e:
        st.error(f"Error fetching historical prices for {ticker}: {str(e)}")
        return None

def get_spy_status_icon_and_label(change_percent: float, timeframe: str) -> Tuple[str, str]:
    """Determine the status icon and label for SPY based on performance and timeframe"""
    # Define thresholds based on timeframe
    thresholds = {
        "5D": (1.5, -1.5),
        "1M": (3.0, -3.0),
        "1Y": (8.0, -8.0),
        "5Y": (35.0, 5.0),
        "10Y": (80.0, 10.0)
    }
    
    up_thresh, down_thresh = thresholds.get(timeframe, (0, 0))
    
    if change_percent >= up_thresh:
        return "🟢", "Bullish"
    elif change_percent <= down_thresh:
        return "🔴", "Bearish"
    else:
        return "🟡", "Neutral"

# Add helper functions for adaptive metrics
def get_metric_labels(period_label):
    """Generate appropriate metric labels based on selected period"""
    label_map = {
        "1D": ("Daily Change %", "Daily Gains/Losses"),
        "5D": ("5-Day Change %", "5-Day Gains/Losses"),
        "1M": ("Monthly Change %", "Monthly Gains/Losses"),
        "6M": ("6-Month Change %", "6-Month Gains/Losses"),
        "YTD": ("YTD Change %", "YTD Gains/Losses"),
        "1Y": ("Annual Change %", "Annual Gains/Losses"),
        "5Y": ("5-Year Change %", "5-Year Gains/Losses"),
        "10Y": ("10-Year Change %", "10-Year Gains/Losses"),
        "MAX": ("All-Time Change %", "All-Time Gains/Losses")
    }
    return label_map.get(period_label, ("Change %", "Gains/Losses"))

def calculate_period_change(close_series, period_label):
    """Calculate percentage and value change with robust period handling"""
    # Validate input
    if close_series is None or not isinstance(close_series, pd.Series):
        return None, None
        
    # Ensure we have a datetime index
    if not isinstance(close_series.index, pd.DatetimeIndex):
        try:
            close_series.index = pd.to_datetime(close_series.index)
        except Exception:
            return None, None
    
    # Clean and sort the data
    clean_series = close_series.dropna().sort_index()
    
    # Check we have enough data
    if len(clean_series) < 2:
        return None, None
    
    try:
        # Handle different period types
        if period_label == "1D":
            # For intraday data, ensure we have today's complete session
            today = pd.Timestamp.now().normalize()
            mask = (clean_series.index >= today)
            clean_series = clean_series[mask]
            
            # Need at least 2 points to calculate change
            if len(clean_series) < 2:
                return None, None
                
            # Use first and last point of today's data
            start_price = float(clean_series.iloc[0])
            end_price = float(clean_series.iloc[-1])
            
        elif period_label == "5D":
            # Get data from last 5 trading days
            five_days_ago = pd.Timestamp.now() - pd.Timedelta(days=5)
            mask = (clean_series.index >= five_days_ago)
            clean_series = clean_series[mask]
            
            if len(clean_series) < 2:
                return None, None
                
            start_price = float(clean_series.iloc[0])
            end_price = float(clean_series.iloc[-1])
            
        else:
            # For all other periods, use first and last available points
            start_price = float(clean_series.iloc[0])
            end_price = float(clean_series.iloc[-1])
        
        # Validate prices
        if start_price <= 0 or end_price <= 0:
            return None, None
            
        # Calculate changes
        pct_change = ((end_price - start_price) / start_price) * 100
        value_change = end_price - start_price
        
        return round(pct_change, 2), round(value_change, 2)
        
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Error in calculate_period_change for {period_label}: {str(e)}")
            st.error(traceback.format_exc())
        return None, None

@st.cache_data(ttl=3600)
def calculate_cached_period_change(ticker, period, interval):
    """Calculate period change with caching"""
    try:
        data = yf.download(to_yahoo_ticker(ticker), period=period, interval=interval)
        if data.empty:
            return None
        return calculate_period_change(data['Close'], period)
    except Exception as e:
        st.error(f"Error calculating period change for {ticker}: {str(e)}")
        return None

# Set page to wide layout
st.set_page_config(
    page_title="Portfolio Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'financial_data_cache' not in st.session_state:
    st.session_state['financial_data_cache'] = {}
if 'portfolio_created' not in st.session_state:
    st.session_state['portfolio_created'] = False
if 'show_charts' not in st.session_state:
    st.session_state['show_charts'] = False
if 'show_financials' not in st.session_state:
    st.session_state['show_financials'] = False
if 'show_monte_carlo' not in st.session_state:
    st.session_state['show_monte_carlo'] = False
if 'show_market_overview' not in st.session_state:
    st.session_state['show_market_overview'] = False
if 'custom_weight_inputs' not in st.session_state:
    st.session_state['custom_weight_inputs'] = {}

# Show initial message if portfolio not created
if not st.session_state.get('portfolio_created'):
    st.info("Enter tickers and click Apply to view your portfolio dashboard.")

# Custom CSS for full width and better spacing
st.markdown("""
<style>
    /* Main container adjustments */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem !important;  /* Increased bottom padding */
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Add metric explanation styling */
    .metric-explanation {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.8);
        margin-top: 2px;
        margin-bottom: 5px;
    }
    
    /* Full width elements */
    div.stButton > button {
        width: 100%;
    }
    div.stDataFrame {
        width: 100%;
    }
    
    /* Section padding */
    section.main > div {
        padding-left: 5%;
        padding-right: 5%;
    }
    
    /* Improve spacing between elements */
    .stMarkdown {
        margin-bottom: 0.5rem;
    }
    
    /* Better table formatting */
    .stDataFrame {
        font-size: 0.9rem;
    }
    
    /* Enhanced metric cards styling */
    div[data-testid="stMetric"] {
        background-color: #1f77b4;
        border-radius: 0.5rem;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out;
        margin-bottom: 1.5rem;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    div[data-testid="stMetric"] > div > div {
        color: white !important;
    }
    
    div[data-testid="stMetric"] > div > div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    
    div[data-testid="stMetric"] > div > div[data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        opacity: 0.9;
    }
    
    /* Consistent chart heights */
    .stPlotlyChart {
        height: 400px;
    }
    
    /* Better form spacing */
    .stForm {
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Improve selectbox and multiselect */
    .stSelectbox, .stMultiSelect {
        margin-bottom: 0.5rem;
    }
    
    /* Better tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    /* Consistent spacing for columns */
    .row-widget.stHorizontal {
        gap: 1rem;
    }
    
    /* Add spacing after sections */
    .stMarkdown h3 {
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f2f6;
    }
    
    /* Improve section headers */
    .stMarkdown h2 {
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        color: #1f77b4;
        font-weight: 600;
    }
    
    /* Add subtle background to sections */
    .stMarkdown h2 + div {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    
    /* Improve spacing between metrics and charts */
    div[data-testid="stMetric"] + div {
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state first, before any other code
if 'ticker_info_cache' not in st.session_state:
    st.session_state['ticker_info_cache'] = {}
if 'kaggle_path' not in st.session_state:
    st.session_state['kaggle_path'] = None
if 'optimal_result' not in st.session_state:
    st.session_state['optimal_result'] = None
if 'optimal_metrics' not in st.session_state:
    st.session_state['optimal_metrics'] = None
if 'optimal_backtest' not in st.session_state:
    st.session_state['optimal_backtest'] = None
if 'optimal_table' not in st.session_state:
    st.session_state['optimal_table'] = None
if 'sector_map' not in st.session_state:
    st.session_state['sector_map'] = None
if 'missing_tickers' not in st.session_state:
    st.session_state['missing_tickers'] = None
if 'selected_tickers' not in st.session_state:
    st.session_state['selected_tickers'] = None
if 'risk_level' not in st.session_state:
    st.session_state['risk_level'] = None
if 'close_prices' not in st.session_state:
    st.session_state['close_prices'] = None
if 'portfolio_weights' not in st.session_state:
    st.session_state['portfolio_weights'] = None
if 'ef_portfolios' not in st.session_state:
    st.session_state['ef_portfolios'] = None

# Finnhub API key (temporary, should be moved to secrets.toml)
finnhub_api_key = "d0r448pr01qn4tjgbsfgd0r448pr01qn4tjgbsg0"

def get_finnhub_top_movers(direction='gainers', count=10):
    """Get top gainers or losers from Finnhub"""
    url = f"https://finnhub.io/api/v1/news?category=general&token={finnhub_api_key}"
    mover_url = "https://finnhub.io/api/v1/scan/technical-indicator"
    response = requests.get("https://finnhub.io/api/v1/stock/symbol?exchange=US&token=" + finnhub_api_key)
    if response.status_code != 200:
        return []

    all_tickers = [x['symbol'] for x in response.json() if x['type'] == 'Common Stock']
    movers = []

    for symbol in all_tickers[:100]:  # limit for testing
        try:
            quote = requests.get(f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={finnhub_api_key}").json()
            change = ((quote["c"] - quote["pc"]) / quote["pc"]) * 100 if quote["pc"] else 0
            movers.append((symbol, change))
        except:
            continue

    sorted_movers = sorted(movers, key=lambda x: x[1], reverse=True)
    if direction == "gainers":
        return sorted_movers[:count]
    else:
        return sorted_movers[-count:][::-1]

def get_finnhub_sector_performance():
    """Get sector performance from Finnhub"""
    url = f"https://finnhub.io/api/v1/stock/sector-performance?token={finnhub_api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        return [(x['sector'], x['performance']) for x in data]
    except:
        return []

@st.cache_data(ttl=86400)  # 24 hours for S&P 500 tickers
def get_sp500_tickers():
    """
    Fetch S&P 500 tickers with sector info:
    1. Try yFinance from ^GSPC components.
    2. Fallback to maintained GitHub CSV.
    Returns DataFrame with Symbol, Name, Sector, Industry columns
    """
    # Try yFinance first
    try:
        sp500 = yf.Ticker("^GSPC")
        components = sp500.components
        if components is not None and not components.empty:
            components.index = components.index.str.replace(".", "-", regex=False)
            components = components.reset_index()
            components.columns = ['Symbol', 'Name']
            # Try to get sector info from yfinance
            sectors = []
            for ticker in components['Symbol']:
                try:
                    info = yf.Ticker(ticker).info
                    sectors.append({
                        'Sector': info.get('sector', 'Unknown'),
                        'Industry': info.get('industry', 'Unknown')
                    })
                except:
                    sectors.append({'Sector': 'Unknown', 'Industry': 'Unknown'})
            sectors_df = pd.DataFrame(sectors)
            return pd.concat([components, sectors_df], axis=1)
    except Exception:
        pass  # Fail silently and try fallback
    
    # Fallback: Use GitHub CSV
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df = pd.read_csv(url)
        df['Symbol'] = df['Symbol'].str.replace(".", "-", regex=False)
        # Rename columns to match yfinance format
        df = df.rename(columns={
            'Security': 'Name',
            'GICS Sector': 'Sector',
            'GICS Sub-Industry': 'Industry'
        })
        return df[['Symbol', 'Name', 'Sector', 'Industry']]
    except Exception as e:
        st.warning(f"Could not fetch S&P 500 tickers: {str(e)}. Using fallback list.")
        fallback = pd.DataFrame({
            'Symbol': ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "PG"],
            'Name': ["Apple", "Microsoft", "Alphabet", "Amazon", "Tesla", "Nvidia", "Meta", "JPMorgan", "Visa", "Procter & Gamble"],
            'Sector': ["Technology"]*6 + ["Communication Services", "Financial Services", "Financial Services", "Consumer Defensive"],
            'Industry': ["Consumer Electronics", "Software", "Internet Content", "E-Commerce", "Auto Manufacturers", "Semiconductors", 
                        "Internet Content", "Banks", "Credit Services", "Household Products"]
        })
        return fallback

@st.cache_data(ttl=3600)  # 1 hour for market data
def get_top_movers(timeframe="1d"):
    """Fetch top gainers/losers from S&P 500 for a timeframe."""
    with st.spinner("Fetching market data. This may take 10-15 seconds..."):
        # Get S&P 500 tickers
        try:
            sp500_tickers = get_sp500_tickers()
            if not sp500_tickers.empty:
                return sp500_tickers
        except Exception as e:
            st.warning(f"Error getting S&P 500 tickers: {str(e)}")
            return pd.DataFrame()
        
        # Limit to top 100 tickers for better performance
        sp500_tickers = sp500_tickers[:100]
        
        # Adjust intervals for better performance
        timeframe_map = {
            "1D": ("1d", "15m"),
            "1W": ("5d", "1h"),
            "1M": ("1mo", "1d"),
            "3M": ("3mo", "1d"),
            "YTD": ("ytd", "1d"),
            "1Y": ("1y", "1wk"),
            "5Y": ("5y", "1mo"),
            "10Y": ("10y", "1mo")
        }
        
        if timeframe in timeframe_map:
            period, interval = timeframe_map[timeframe]
        else:
            period, interval = "1mo", "1d"
            
        try:
            # Download data with MultiIndex structure and threading enabled
            data = yf.download(
                sp500_tickers['Symbol'], 
                period=period, 
                interval=interval, 
                group_by="ticker", 
                progress=False, 
                threads=True
            )
            
            returns = {}
            failed_tickers = []
            
            for ticker in sp500_tickers['Symbol']:
                try:
                    # Handle both MultiIndex and flat index structures
                    if isinstance(data.columns, pd.MultiIndex):
                        if ('Close', ticker) in data.columns:
                            close = data['Close'][ticker]
                        else:
                            continue
                    else:
                        if ticker in data.columns:
                            close = data[ticker]
                        else:
                            continue
                    
                    # Fill missing values and ensure we have enough data
                    close = close.ffill().bfill()
                    if len(close) > 1:
                        start_price = close.iloc[0]
                        end_price = close.iloc[-1]
                        if start_price > 0 and end_price > 0:  # Additional validation
                            returns[ticker] = (end_price - start_price) / start_price * 100
                except Exception as e:
                    failed_tickers.append(ticker)
                    continue
                    
            if failed_tickers:
                st.warning(f"Could not fetch data for {len(failed_tickers)} tickers")
                
            if not returns:
                st.warning("No valid return data found for selected timeframe.")
                return pd.DataFrame()
                
            sorted_returns = sorted(returns.items(), key=lambda x: x[1], reverse=True)
            return sp500_tickers.iloc[sorted_returns[:10].index]
            
        except Exception as e:
            st.error(f"Error downloading market data: {str(e)}")
            return pd.DataFrame()

@st.cache_data(ttl=3600)  # 1 hour for sector performance
def get_sector_performance_custom(timeframe="1mo"):
    """Fetch sector performance using sector ETFs with custom timeframe support."""
    sectors = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Financials': 'XLF',
        'Consumer Discretionary': 'XLY',
        'Industrials': 'XLI',
        'Energy': 'XLE',
        'Consumer Staples': 'XLP',
        'Utilities': 'XLU',
        'Materials': 'XLB',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC'
    }

    # Map custom timeframes to yfinance parameters
    timeframe_map = {
        "This Week": ("5d", "1d"),
        "This Month": ("1mo", "1d"),
        "This Year": ("1y", "1wk"),
        "YTD": ("ytd", "1d"),
        "5Y": ("5y", "1mo"),
        "10Y": ("10y", "1mo")
    }
    
    if timeframe not in timeframe_map:
        period, interval = "1mo", "1d"
    else:
        period, interval = timeframe_map[timeframe]

    try:
        # Download data for each sector ETF individually
        sector_perf = []
        for name, symbol in sectors.items():
            try:
                # Download data for this ETF
                data = yf.download(
                    symbol,
                    period=period,
                    interval=interval,
                    progress=False
                )
                
                if data.empty:
                    st.warning(f"No data returned for {name} ({symbol})")
                    continue
                
                # Get close prices - ensure we get a Series, not a DataFrame
                close_prices = data['Close'].dropna() if 'Close' in data.columns else None
                
                if close_prices is None or len(close_prices) < 2:
                    st.warning(f"Insufficient data points for {name} ({symbol})")
                    continue
                
                # Convert to numpy array if needed to avoid Series comparison issues
                close_values = close_prices.values if hasattr(close_prices, 'values') else close_prices
                
                # Calculate performance - ensure we're working with scalar values
                start_price = float(close_values[0])
                end_price = float(close_values[-1])
                
                # Validate prices
                if start_price <= 0 or end_price <= 0:
                    st.warning(f"Invalid prices for {name} ({symbol}): start={start_price}, end={end_price}")
                    continue
                
                perf = (end_price - start_price) / start_price * 100
                
                # Validate performance calculation
                if not isinstance(perf, Number) or np.isnan(perf) or np.isinf(perf):
                    st.warning(f"Invalid performance calculation for {name} ({symbol}): {perf}")
                    continue
                
                sector_perf.append((name, float(perf)))
                
            except Exception as e:
                st.warning(f"Error processing {name} ({symbol}): {str(e)}")
                continue
        
        if not sector_perf:
            st.warning("No valid sector performance data could be calculated")
            return []
            
        # Sort by performance (highest first)
        return sorted(sector_perf, key=lambda x: x[1], reverse=True)
        
    except Exception as e:
        st.error(f"Error fetching sector performance: {str(e)}")
        if st.session_state.get('debug_mode', False):
            st.error(traceback.format_exc())
        return []

# --- Key Financial Ratios Helper Functions ---
def horizontal_analysis(prev_value, current_value):
    """Calculate percentage change between two values (horizontal analysis)"""
    if prev_value is None or current_value is None or prev_value == 0:
        return None
    return ((current_value - prev_value) / prev_value) * 100

def current_ratio(current_assets, current_liabilities):
    return current_assets / current_liabilities if current_liabilities else None

def quick_ratio(current_assets, inventory, current_liabilities):
    return (current_assets - inventory) / current_liabilities if current_liabilities else None

def working_capital(current_assets, current_liabilities):
    return current_assets - current_liabilities if current_assets is not None and current_liabilities is not None else None

def gross_margin(gross_profit, revenue):
    return gross_profit / revenue if revenue else None

def operating_margin(operating_income, revenue):
    return operating_income / revenue if revenue else None

def net_margin(net_income, revenue):
    return net_income / revenue if revenue else None

def return_on_assets(net_income, total_assets):
    return net_income / total_assets if total_assets else None

def return_on_equity(net_income, equity):
    return net_income / equity if equity else None

def dso(avg_receivables, revenue):
    return (avg_receivables / revenue) * 365 if avg_receivables is not None and revenue else None

def dpo(avg_payables, purchases):
    return (avg_payables / purchases) * 360 if avg_payables is not None and purchases else None

def dio(avg_inventory, cogs):
    return (avg_inventory / cogs) * 365 if avg_inventory is not None and cogs else None

def cash_conversion_cycle(dso_val, dio_val, dpo_val):
    if dso_val is not None and dio_val is not None and dpo_val is not None:
        return dso_val + dio_val - dpo_val
    return None

def interpret_ratios(ratios):
    insights = []

    def fmt(val, percent=False):
        try:
            if val is None:
                return "N/A"
            if percent:
                return f"{val:.2%}"
            else:
                return f"{val:,.2f}"
        except:
            return "N/A"

    # --- Liquidity ---
    cr = ratios.get("Current Ratio")
    if isinstance(cr, (int, float)):
        if cr < 1:
            insights.append(f"🔴 **Current Ratio < 1** — may struggle to meet obligations. ({fmt(cr)})")
        elif cr > 3:
            insights.append(f"🟡 **Current Ratio > 3** — may suggest excess idle assets. ({fmt(cr)})")
        else:
            insights.append(f"🟢 **Current Ratio** is healthy. ({fmt(cr)})")

    qr = ratios.get("Quick Ratio")
    if isinstance(qr, (int, float)):
        if qr < 1:
            insights.append(f"🔴 **Quick Ratio < 1** — liquidity concern without inventory. ({fmt(qr)})")
        elif qr < 2:
            insights.append(f"🟡 **Quick Ratio** between 1–2 — moderate liquidity. ({fmt(qr)})")
        else:
            insights.append(f"🟢 **Quick Ratio** is strong. ({fmt(qr)})")

    wc = ratios.get("Working Capital")
    if isinstance(wc, (int, float)):
        if wc < 0:
            insights.append(f"🔴 **Negative Working Capital** — liquidity risk. ({fmt(wc)})")
        else:
            insights.append(f"🟢 **Working Capital** is positive — short-term solvency is fine. ({fmt(wc)})")

    # --- Profitability ---
    gm = ratios.get("Gross Margin")
    if isinstance(gm, (int, float)):
        if gm < 0.2:
            insights.append(f"🔴 **Low Gross Margin** — high production costs. ({fmt(gm, True)})")
        elif gm < 0.4:
            insights.append(f"🟡 **Gross Margin** is moderate — average cost control. ({fmt(gm, True)})")
        else:
            insights.append(f"🟢 **Gross Margin** is strong — efficient cost structure. ({fmt(gm, True)})")

    om = ratios.get("Operating Margin")
    if isinstance(om, (int, float)):
        if om < 0.1:
            insights.append(f"🔴 **Low Operating Margin** — heavy operating costs. ({fmt(om, True)})")
        elif om < 0.2:
            insights.append(f"🟡 **Operating Margin** is moderate — room for efficiency. ({fmt(om, True)})")
        else:
            insights.append(f"🟢 **Operating Margin** is healthy. ({fmt(om, True)})")

    nm = ratios.get("Net Margin")
    if isinstance(nm, (int, float)):
        if nm < 0:
            insights.append(f"🔴 **Negative Net Margin** — company is unprofitable. ({fmt(nm, True)})")
        elif nm > 0.20:
            insights.append(f"🟢 **Net Margin** is excellent — highly profitable. ({fmt(nm, True)})")
        else:
            insights.append(f"🟡 **Net Margin** is moderate. ({fmt(nm, True)})")

    roa = ratios.get("ROA")
    if isinstance(roa, (int, float)):
        if roa < 0:
            insights.append(f"🔴 **Negative ROA** — inefficient asset use. ({fmt(roa, True)})")
        elif roa > 0.1:
            insights.append(f"🟢 **ROA** is strong — assets generate solid returns. ({fmt(roa, True)})")
        else:
            insights.append(f"🟡 **ROA** is average. ({fmt(roa, True)})")

    roe = ratios.get("ROE")
    if isinstance(roe, (int, float)):
        if roe < 0:
            insights.append(f"🔴 **Negative ROE** — equity is being eroded. ({fmt(roe, True)})")
        elif roe > 0.15:
            insights.append(f"🟢 **ROE** is high — strong returns to shareholders. ({fmt(roe, True)})")
        else:
            insights.append(f"🟡 **ROE** is average. ({fmt(roe, True)})")

    # --- Efficiency ---
    dso = ratios.get("DSO")
    if isinstance(dso, (int, float)):
        if dso > 60:
            insights.append(f"🔴 **High DSO** — slow collection from customers. ({fmt(dso)})")
        elif dso < 30:
            insights.append(f"🟢 **Low DSO** — fast payment cycle. ({fmt(dso)})")
        else:
            insights.append(f"🟡 **DSO** is average. ({fmt(dso)})")

    dpo = ratios.get("DPO")
    if isinstance(dpo, (int, float)):
        if dpo > 90:
            insights.append(f"🟢 **High DPO** — good supplier credit use. ({fmt(dpo)})")
        elif dpo < 30:
            insights.append(f"🔴 **Low DPO** — short supplier payment terms. ({fmt(dpo)})")
        else:
            insights.append(f"🟡 **DPO** is within normal range. ({fmt(dpo)})")

    dio = ratios.get("DIO")
    if isinstance(dio, (int, float)):
        if dio > 90:
            insights.append(f"🔴 **High DIO** — slow inventory turnover. ({fmt(dio)})")
        elif dio < 30:
            insights.append(f"🟢 **Low DIO** — inventory moves fast. ({fmt(dio)})")
        else:
            insights.append(f"🟡 **DIO** is average. ({fmt(dio)})")

    ccc = ratios.get("Cash Conversion Cycle")
    if isinstance(ccc, (int, float)):
        if ccc < 0:
            insights.append(f"🟢 **Negative CCC** — efficient cash cycle. ({fmt(ccc)})")
        elif ccc < 60:
            insights.append(f"🟡 **CCC** is reasonable. ({fmt(ccc)})")
        else:
            insights.append(f"🔴 **High CCC** — slow liquidity conversion. ({fmt(ccc)})")

    return insights

@st.cache_data(ttl=3600)  # 1 hour for asset price changes
def get_asset_price_change(symbol: str, period: str = "1mo"):
    """Get price change for an asset"""
    try:
        data = yf.download(to_yahoo_ticker(symbol), period=period)
        if data.empty:
            return None
        return calculate_period_change(data['Close'], period)
    except Exception as e:
        st.error(f"Error getting price change for {symbol}: {str(e)}")
        return None

# Monte Carlo Simulation Helper
def monte_carlo_simulation(start_value, mean_return, volatility, years=10, simulations=500, steps_per_year=252):
    dt = 1 / steps_per_year
    total_steps = int(steps_per_year * years)
    results = np.zeros((simulations, total_steps))
    for i in range(simulations):
        prices = [start_value]
        for _ in range(total_steps - 1):
            shock = np.random.normal(loc=(mean_return - 0.5 * volatility**2) * dt, scale=volatility * np.sqrt(dt))
            prices.append(prices[-1] * np.exp(shock))
        results[i] = prices
    return results

def calculate_annual_returns(close_prices, weights):
    """Calculate annual returns for the portfolio"""
    # Resample to yearly closing prices
    yearly_prices = close_prices.resample('Y').last()
    # Calculate yearly returns
    yearly_returns = yearly_prices.pct_change().dropna()
    # Calculate weighted portfolio returns
    port_returns = (yearly_returns * weights).sum(axis=1)
    return port_returns

# Inject tooltip CSS at the very top, only once
if 'custom_tooltip_css' not in st.session_state:
    st.markdown(
        """
        <style>
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 260px;
            background-color: #222;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 8px 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -130px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.9em;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.session_state['custom_tooltip_css'] = True

# Sidebar
# st.sidebar.title("Portfolio Options")

# Global constants
BLUE_PALETTE = ['#1f77b4', '#2ca9e1', '#4dabf7', '#74c0fc', '#90d8fd', '#a5d8ff', '#c4e0ff', '#d6e9ff']

# Initialize ticker info cache in session state
if 'ticker_info_cache' not in st.session_state:
    st.session_state['ticker_info_cache'] = {}

# Initialize financial data cache
if 'financial_data_cache' not in st.session_state:
    st.session_state['financial_data_cache'] = {}

# Persist weights across reruns
if 'custom_weight_inputs' not in st.session_state:
    st.session_state['custom_weight_inputs'] = {}

def get_info(ticker):
    """Get ticker info with caching"""
    # Ensure cache exists
    if 'ticker_info_cache' not in st.session_state:
        st.session_state['ticker_info_cache'] = {}
    
    cache = st.session_state['ticker_info_cache']
    
    if ticker not in cache:
        try:
            with st.spinner(f"Fetching data for {ticker}..."):
                cache[ticker] = yf.Ticker(to_yahoo_ticker(ticker)).info or {}
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            cache[ticker] = {}
    
    return cache[ticker]

def fetch_ticker_info_parallel(tickers):
    """Fetch ticker info in parallel using ThreadPoolExecutor"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(tickers))) as executor:
        future_to_ticker = {executor.submit(get_info, ticker): ticker for ticker in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                future.result()  # This will raise any exceptions that occurred
            except Exception as e:
                st.warning(f"Error fetching data for {ticker}: {str(e)}")

@lru_cache(maxsize=100)
def get_financial_data(ticker, statement_type):
    """Get financial data with caching"""
    cache_key = f"{ticker}_{statement_type}"
    if cache_key not in st.session_state['financial_data_cache']:
        try:
            ticker_obj = yf.Ticker(to_yahoo_ticker(ticker))
            if statement_type == 'income':
                data = ticker_obj.income_stmt
            elif statement_type == 'balance':
                data = ticker_obj.balance_sheet
            elif statement_type == 'cashflow':
                data = ticker_obj.cashflow
            else:
                return None
            st.session_state['financial_data_cache'][cache_key] = data
        except Exception:
            st.session_state['financial_data_cache'][cache_key] = None
    return st.session_state['financial_data_cache'][cache_key]

# Helper functions at the top
@st.cache_data(ttl=86400)  # 24 hours for financial data
def get_sector_allocation_from_yfinance(tickers):
    """Get sector allocation using yfinance and fallback to GitHub data"""
    # Get the full S&P 500 data
    sp500_data = get_sp500_tickers()
    
    sector_map = {}
    missing_tickers = []
    
    for ticker in tickers:
        # First try yfinance
        try:
            info = yf.Ticker(to_yahoo_ticker(ticker)).info
            sector = info.get("sector", None)
            if sector:
                sector_map[ticker] = sector
                continue  # Found in yfinance, move to next ticker
        except Exception:
            pass
        
        # Fallback to GitHub data
        ticker_row = sp500_data[sp500_data['Symbol'].str.upper() == ticker.upper()]
        if not ticker_row.empty:
            sector = ticker_row['Sector'].values[0]
            if sector != 'Unknown':
                sector_map[ticker] = sector
                continue
        
        # If we get here, we couldn't find the sector
        missing_tickers.append(ticker)
    
    return sector_map, missing_tickers

def to_yahoo_ticker(ticker):
    """Convert ticker symbol to Yahoo Finance format"""
    return ticker.replace('.', '-')

def format_market_cap(val):
    try:
        val = float(val)
        if val >= 1e12:
            return f"${val/1e12:.2f}T"
        elif val >= 1e9:
            return f"${val/1e9:.2f}B"
        elif val >= 1e6:
            return f"${val/1e6:.2f}M"
        elif val >= 1e3:
            return f"${val/1e3:.2f}K"
        else:
            return f"${val:,.0f}"
    except Exception:
        return 'N/A'

# Cache and reuse all_tickers
all_tickers = get_sp500_tickers()

# Period selector (now above charts, not in sidebar)
period_options = {
    "1D": ("1d", "15m"),  # Changed interval from 5m to 15m for 1D
    "5D": ("5d", "5m"),   # Changed interval from 15m to 5m for 5D
    "1M": ("1mo", "30m"),
    "6M": ("6mo", "1d"),
    "YTD": ("ytd", "1d"),
    "1Y": ("1y", "1d"),
    "5Y": ("5y", "1wk"),
    "10Y": ("10y", "1mo"),
    "MAX": ("max", "1mo")
}

st.title("Portfolio Dashboard")

# --- UI Reorganization ---
# 1. Stock Tickers and Risk Level at the top
st.markdown("## Create Optimal Portfolio")

# Tooltip CSS (keep this at the top or before usage)
st.markdown("""
<style>
    .custom-tooltip {
        display: inline-block;
        margin-left: 5px;
        color: #1f77b4;
        cursor: help;
    }
    .custom-tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: #333;
        color: white;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9em;
    }
    .custom-tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

risk_level = st.radio(
    "Select Risk Level", 
    ["Low", "Moderate", "High", "Custom"], 
    horizontal=True,
    help="Choose how aggressively your portfolio should be optimized."
)

# Risk descriptions with inline tooltip icon
risk_descriptions = {
    "Low": "<strong>Low</strong>: Prioritizes capital preservation with minimal volatility. <span class='custom-tooltip'>ℹ️<span class='tooltiptext'><strong>Low</strong>: Minimizes volatility (reduces risk). Best for conservative investors.<br><br><strong>Moderate</strong>: Balances risk/return (maximizes Sharpe Ratio).<br><br><strong>High</strong>: Maximizes returns (ignores volatility).<br><br><strong>Custom</strong>: Set your own weights manually.</span></span>",
    "Moderate": "<strong>Moderate</strong>: Balances risk and return for optimal growth. <span class='custom-tooltip'>ℹ️<span class='tooltiptext'><strong>Low</strong>: Minimizes volatility (reduces risk). Best for conservative investors.<br><br><strong>Moderate</strong>: Balances risk/return (maximizes Sharpe Ratio).<br><br><strong>High</strong>: Maximizes returns (ignores volatility).<br><br><strong>Custom</strong>: Set your own weights manually.</span></span>",
    "High": "<strong>High</strong>: Targets maximum returns (highest volatility). <span class='custom-tooltip'>ℹ️<span class='tooltiptext'><strong>Low</strong>: Minimizes volatility (reduces risk). Best for conservative investors.<br><br><strong>Moderate</strong>: Balances risk/return (maximizes Sharpe Ratio).<br><br><strong>High</strong>: Maximizes returns (ignores volatility).<br><br><strong>Custom</strong>: Set your own weights manually.</span></span>",
    "Custom": "<strong>Custom</strong>: Manually allocate weights to each asset. <span class='custom-tooltip'>ℹ️<span class='tooltiptext'><strong>Low</strong>: Minimizes volatility (reduces risk). Best for conservative investors.<br><br><strong>Moderate</strong>: Balances risk/return (maximizes Sharpe Ratio).<br><br><strong>High</strong>: Maximizes returns (ignores volatility).<br><br><strong>Custom</strong>: Set your own weights manually.</span></span>"
}
st.markdown(risk_descriptions[risk_level], unsafe_allow_html=True)

# Add Start Year and End Year dropdowns (max 15 years ago)
current_year = datetime.now().year
max_years = 15
year_options = list(range(current_year, current_year - max_years, -1))
col1, col2 = st.columns(2)
with col1:
    start_year = st.selectbox(
        "Start Year ",
        options=year_options[::-1],  # Ascending order for start
        index=len(year_options) - 1,
        help="Select the starting year for your analysis (max 15 years ago)."
    )
with col2:
    end_year = st.selectbox(
        "End Year ",
        options=year_options,
        index=0,
        help="Select the ending year for your analysis."
    )
if start_year > end_year:
    st.warning("Start Year must be less than or equal to End Year.")
    st.stop()
# Store in session state for later use
st.session_state['start_year'] = start_year
st.session_state['end_year'] = end_year

# Inside the form only include tickers and button
with st.form("portfolio_input_form"):
    selected_tickers = st.multiselect(
        "Stock Tickers",
        options=all_tickers,
        default=["AAPL", "TSLA", "MSFT"],
        help="Start typing to search and select tickers from the S&P 500."
    )
    custom_weights = {}
    total_weight = 0
    if risk_level == "Custom" and selected_tickers:
        st.markdown("### Manual Allocation")
        # Remove weights for unselected tickers
        for t in list(st.session_state['custom_weight_inputs'].keys()):
            if t not in selected_tickers:
                del st.session_state['custom_weight_inputs'][t]
        # Add new tickers with default 0.0
        for t in selected_tickers:
            if t not in st.session_state['custom_weight_inputs']:
                st.session_state['custom_weight_inputs'][t] = 0.0
        # Input fields and update session state
        for ticker in selected_tickers:
            weight = st.number_input(
                f"Weight for {ticker} (%)",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                format="%.2f",
                value=st.session_state['custom_weight_inputs'][ticker],
                key=f"weight_{ticker}"
            )
            st.session_state['custom_weight_inputs'][ticker] = weight
        # Calculate total weight from session state
        total_weight = sum(st.session_state['custom_weight_inputs'][t] for t in selected_tickers)
        st.markdown(f"**Total Allocation: {total_weight:.2f}%**")
        if abs(total_weight - 100.0) > 0.01:
            st.warning("Total allocation must be exactly 100%. Adjust the weights above.")
    create_portfolio = st.form_submit_button(
        "Update Portfolio" if st.session_state.get('portfolio_created') else "Create Portfolio"
    )

def get_optimized_weights(mean_returns, cov_matrix, risk_level):
    """Get optimized portfolio weights based on risk level"""
    # Validate inputs
    if not isinstance(mean_returns, (pd.Series, np.ndarray)) or len(mean_returns) == 0:
        st.error("Invalid mean returns input")
        return None
        
    if not isinstance(cov_matrix, (pd.DataFrame, np.ndarray)) or cov_matrix.size == 0:
        st.error("Invalid covariance matrix input")
        return None
        
    n_assets = len(mean_returns)
    
    # Check for NaN values
    if np.isnan(mean_returns).any() or np.isnan(cov_matrix).any():
        st.error("NaN values detected in inputs")
        return np.array([1./n_assets] * n_assets)  # Return equal weights
    
    # Define constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
    ]
    bounds = tuple((0, 1) for _ in range(n_assets))  # weights between 0 and 1
    
    # Initial guess (equal weights)
    initial_weights = np.array([1./n_assets] * n_assets)
    
    # Define objective function based on risk level
    if risk_level == "Low":
        # Minimize volatility
        objective = lambda x: portfolio_volatility(x, mean_returns, cov_matrix)
    elif risk_level == "High":
        # Maximize returns (minimize negative returns)
        objective = lambda x: -portfolio_return(x, mean_returns)
    else:  # Moderate
        # Maximize Sharpe ratio (minimize negative Sharpe)
        objective = lambda x: negative_sharpe(x, mean_returns, cov_matrix)
    
    # Optimize
    try:
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            st.warning(f"Optimization did not converge: {result.message}")
            return initial_weights
            
        return result.x
        
    except Exception as e:
        st.error(f"Error during optimization: {str(e)}")
        return initial_weights

if create_portfolio and selected_tickers:
    # Convert years to date strings for yfinance
    start_date = f"{st.session_state['start_year']}-01-01"
    end_date = f"{st.session_state['end_year']}-12-31"
    
    if risk_level == "Custom":
        # Use custom weights from session state
        custom_weights = {t: st.session_state['custom_weight_inputs'].get(t, 0.0) for t in selected_tickers}
        total_weight = sum(custom_weights[t] for t in selected_tickers)
        if abs(total_weight - 100.0) > 0.01:
            st.error("Custom allocation must sum to exactly 100%. Please adjust your weights.")
            st.stop()
        weights = np.array([custom_weights[t] / 100 for t in selected_tickers])
        valid_tickers = selected_tickers
    else:
        n = len(selected_tickers)
        # Download price data with improved error handling
        with st.spinner("Downloading price data for optimization..."):
            valid_tickers = []
            failed_tickers = []
            close_prices = []
            
            # Download each ticker individually with error handling
            for ticker in selected_tickers:
                try:
                    yahoo_ticker = to_yahoo_ticker(ticker)
                    data = yf.download(yahoo_ticker, start=start_date, end=end_date, progress=False)
                    if data.empty or 'Close' not in data.columns:
                        failed_tickers.append(ticker)
                        continue
                        
                    close_prices.append(data['Close'].rename(ticker))
                    valid_tickers.append(ticker)
                except Exception as e:
                    failed_tickers.append(ticker)
                    continue
            
            if not close_prices:
                st.error("No valid data could be downloaded for any ticker")
                st.stop()
                
            if failed_tickers:
                st.warning(f"Failed to download data for: {', '.join(failed_tickers)}")
                
            # Combine all valid close prices
            close_prices = pd.concat(close_prices, axis=1)
            
            # Calculate returns and covariance
            try:
                returns = close_prices.pct_change().dropna()
                mean_returns = returns.mean()
                cov_matrix = returns.cov()
                
                # Add debug information
                if st.session_state.get('debug_mode', False):
                    st.write("Debug Info - Mean Returns:", mean_returns)
                    st.write("Debug Info - Covariance Matrix:", cov_matrix)
                    st.write("Debug Info - Close Prices:", close_prices.head())
                    st.write("Debug Info - Returns:", returns.head())
                
                if mean_returns.empty or cov_matrix.empty:
                    st.error("Optimization failed due to insufficient data:")
                    if mean_returns.empty:
                        st.error("- No return data calculated")
                    if cov_matrix.empty:
                        st.error("- No covariance matrix calculated")
                    
                    # Show which tickers have data
                    available_data = []
                    for t in valid_tickers:
                        try:
                            if t in close_prices.columns:
                                available_data.append(f"{t}: {len(close_prices[t].dropna())} data points")
                            else:
                                available_data.append(f"{t}: No data")
                        except:
                            available_data.append(f"{t}: Error checking")
                    
                    st.error("Data availability: " + ", ".join(available_data))
                    st.stop()
                
                # Continue with optimization using valid_tickers
                weights = get_optimized_weights(mean_returns, cov_matrix, risk_level)
                
                if weights is None:
                    st.error("Failed to calculate optimal weights")
                    st.stop()
                
                # Create portfolio DataFrame with only valid tickers
                portfolio = pd.DataFrame({
                    'Ticker': valid_tickers,
                    'Weight': weights
                })
                
                # Display results
                st.write("### Optimized Portfolio")
                st.write(portfolio)
                
                # Download data for backtesting with proper ticker conversion
                try:
                    backtest_data = yf.download(
                        [to_yahoo_ticker(t) for t in valid_tickers] + ["SPY"],
                        start=start_date,
                        end=end_date,
                        interval="1d",
                        threads=True
                    )
                    
                    if backtest_data.empty:
                        st.error("Failed to download backtest data")
                        st.stop()
                    
                    # Store the data for later use
                    st.session_state['backtest_data'] = backtest_data
                    
                except Exception as e:
                    st.error(f"Error downloading backtest data: {str(e)}")
                    st.error(traceback.format_exc())
                    st.stop()
                
            except Exception as e:
                st.error(f"Error calculating returns and covariance: {str(e)}")
                st.error(traceback.format_exc())
                st.stop()

        # Check each ticker individually
        st.write("Verifying data for each ticker...")
        for ticker in selected_tickers:
            try:
                ticker_data = yf.download(ticker, start=start_date, end=end_date)
                if ticker_data.empty:
                    st.warning(f"No data found for {ticker} from {start_date} to {end_date}")
                else:
                    st.success(f"{ticker} has {len(ticker_data)} data points")
            except Exception as e:
                st.error(f"Error checking {ticker}: {str(e)}")

        # Calculate returns and covariance with improved error handling
        try:
            close_prices = data['Adj Close']
            returns = close_prices.pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            
            if mean_returns.empty or cov_matrix.empty:
                st.error("Optimization failed due to insufficient data:")
                if mean_returns.empty:
                    st.error("- No return data calculated")
                if cov_matrix.empty:
                    st.error("- No covariance matrix calculated")
                
                # Show which tickers have data
                available_data = []
                for t in selected_tickers:
                    try:
                        if t in close_prices.columns:
                            available_data.append(f"{t}: {len(close_prices[t].dropna())} data points")
                        else:
                            available_data.append(f"{t}: No data")
                    except:
                        available_data.append(f"{t}: Error checking")
                
                st.error("Data availability: " + ", ".join(available_data))
                st.stop()
                
        except Exception as e:
            st.error(f"Error calculating returns and covariance: {str(e)}")
            st.error(traceback.format_exc())
            st.stop()
        
        weights = get_optimized_weights(mean_returns.values, cov_matrix.values, risk_level)
        custom_weights = {t: w * 100 for t, w in zip(selected_tickers, weights)}
    tickers = selected_tickers
    allocation = {tickers[i]: weights[i] * 100 for i in range(len(tickers))}
    st.session_state['optimal_result'] = allocation
    st.session_state['portfolio_created'] = True
    st.session_state['selected_tickers'] = selected_tickers
    st.session_state['risk_level'] = risk_level
    # Store close_prices and weights in session state for later use
    st.session_state['close_prices'] = close_prices
    st.session_state['portfolio_weights'] = weights
    # Trigger portfolio calculation
    try:
        with st.spinner("Downloading price data..."):
            data = yf.download(tickers + ["SPY"], start=start_date, end=end_date, interval="1d", threads=True)
        
        # Fetch ticker info in parallel
        with st.spinner("Fetching ticker information..."):
            fetch_ticker_info_parallel(tickers)
            
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = pd.DataFrame({t: data['Close', t] for t in tickers if ('Close', t) in data.columns})
            spy_prices = data['Close', 'SPY'] if ('Close', 'SPY') in data.columns else None
        else:
            close_prices = pd.DataFrame({tickers[0]: data['Close']})
            spy_prices = None
        close_prices = close_prices.dropna(axis=1, how='all').ffill().bfill()
        returns = close_prices.pct_change().dropna()
        mean_returns = returns.mean()
        volatilities = returns.std()
        cov_matrix = returns.cov()
        expected_return = np.dot(weights, mean_returns) * 252
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0
        st.session_state['optimal_metrics'] = {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }
        latest_prices = []
        betas = []
        expected_returns = []
        volatilities = []
        for i, t in enumerate(tickers):
            try:
                info = get_info(t)
                price = info.get('regularMarketPrice', np.nan)
                beta = info.get('beta', np.nan)
                # Calculate annualized volatility for each ticker
                try:
                    ticker_returns = returns[t]
                    ann_vol = ticker_returns.std() * np.sqrt(252)
                except Exception:
                    ann_vol = np.nan
                latest_prices.append(price)
                betas.append(beta)
                expected_returns.append(mean_returns[t] * 252)
                volatilities.append(ann_vol)
            except Exception:
                latest_prices.append(np.nan)
                betas.append(np.nan)
                expected_returns.append(np.nan)
                volatilities.append(np.nan)
        table_df = pd.DataFrame({
            'Ticker': tickers,
            'Weight %': [f"{w*100:.2f}%" for w in weights],
            'Latest Price': latest_prices,
            'Exp. Ann. Return': [f"{r:.2%}" for r in expected_returns],
            'Volatility': [f"{v:.2%}" if pd.notna(v) else 'N/A' for v in volatilities],
            'Beta': betas
        })
        # Format Latest Price column with currency
        table_df['Latest Price'] = table_df['Latest Price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else 'N/A')
        st.session_state['optimal_table'] = table_df
        if not close_prices.empty:
            port_returns = (returns * weights).sum(axis=1)
            port_cum = (1 + port_returns).cumprod()
            backtest_df = pd.DataFrame({'Portfolio': port_cum})
            if spy_prices is not None:
                spy_returns = spy_prices.pct_change().dropna()
                spy_cum = (1 + spy_returns).cumprod()
                backtest_df['SPY'] = spy_cum
            st.session_state['optimal_backtest'] = backtest_df
        with st.spinner("Fetching sector data..."):
            sector_map, missing_tickers = get_sector_allocation_from_yfinance(list(allocation.keys()))
            st.session_state['sector_map'] = sector_map
            st.session_state['missing_tickers'] = missing_tickers
    except Exception as e:
        st.warning(f"Error computing optimal portfolio: {e}")
        st.text(traceback.format_exc())
        st.session_state['optimal_result'] = None
        st.session_state['optimal_metrics'] = None
        st.session_state['optimal_backtest'] = None
        st.session_state['optimal_table'] = None
        st.session_state['sector_map'] = None
        st.session_state['missing_tickers'] = None

# After portfolio creation, render Optimal Portfolio section
if st.session_state.get('portfolio_created'):
    if st.session_state.get('optimal_result'):
        metrics = st.session_state.get('optimal_metrics', {})
        st.markdown("### Optimal Portfolio Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Annual Return", f"{metrics.get('expected_return', 0):.2%}")
        m2.metric("Volatility (Risk)", f"{metrics.get('volatility', 0):.2%}")
        m3.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        st.markdown("### Optimal Portfolio Allocation")
        alloc_col, sector_col = st.columns(2)
        with alloc_col:
            alloc = st.session_state['optimal_result']
            
            # Format values for better display
            formatted_values = [round(val, 2) for val in list(alloc.values())]
            
            fig = go.Figure(data=[go.Pie(
                labels=list(alloc.keys()), 
                values=formatted_values, 
                hole=0.3,
                hoverinfo='label+percent+value',
                texttemplate='%{label}: %{percent:.1%}',
                marker=dict(
                    colors=BLUE_PALETTE,
                    line=dict(color='white', width=2)
                )
            )])
            
            fig.update_traces(
                textposition='inside',
                textfont_size=12
            )
            
            fig.update_layout(
                title="Portfolio Allocation",
                title_font_size=18,
                margin=dict(l=20, r=20, t=40, b=20), 
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2)
            )
            
            st.plotly_chart(fig, use_container_width=True, key="alloc_pie")
        with sector_col:
            sector_map, missing_tickers = get_sector_allocation_from_yfinance(list(st.session_state['optimal_result'].keys()))
            if sector_map:
                # Calculate sector weights based on portfolio allocation
                sector_weights = {}
                for ticker, weight in st.session_state['optimal_result'].items():
                    sector = sector_map.get(ticker)
                    if sector:
                        sector_weights[sector] = sector_weights.get(sector, 0) + weight
                
                fig_sector = px.pie(
                    names=list(sector_weights.keys()),
                    values=list(sector_weights.values()),
                    title="Sector Diversification (Weighted)",
                    hole=0.3,
                    color_discrete_sequence=BLUE_PALETTE
                )
                
                fig_sector.update_traces(
                    textposition='inside',
                    texttemplate='%{label}: %{percent:.1%}',
                    textfont_size=12,
                    marker=dict(line=dict(color='white', width=2))
                )
                
                fig_sector.update_layout(
                    title_font_size=18,
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2)
                )
                
                st.plotly_chart(fig_sector, use_container_width=True, key="sector_pie")
            else:
                st.warning("No sector data available from yfinance for the selected tickers.")
            if missing_tickers:
                st.info(f"Could not retrieve sector data for: {missing_tickers}")
        st.markdown("### Allocation Table")
        table_df = st.session_state['optimal_table']
        if 'Beta' in table_df:
            def format_beta(val):
                try:
                    return f"{val:.2f}" if isinstance(val, (int, float)) else 'N/A'
                except Exception:
                    return 'N/A'
            table_df['Beta'] = table_df['Beta'].apply(format_beta)
            # Keep Beta column name simple
            table_df.rename(columns={'Beta': 'Beta'}, inplace=True)
        st.dataframe(table_df.style.set_properties(**{'font-size': '12px'}), use_container_width=True)
        if st.session_state.get('optimal_backtest') is not None:
            st.markdown("### Backtest: Cumulative Returns vs. SPY")
            backtest_df = st.session_state['optimal_backtest']
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Portfolio'], mode='lines', name='Portfolio', line=dict(color=BLUE_PALETTE[0])))
            if 'SPY' in backtest_df:
                fig2.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['SPY'], mode='lines', name='SPY', line=dict(color=BLUE_PALETTE[1])))
            fig2.update_layout(xaxis_title="Date", yaxis_title="Cumulative Return", margin=dict(l=20, r=20, t=40, b=20), height=400)
            st.plotly_chart(fig2, use_container_width=True)

            # --- Annual Returns Bar Chart ---
            st.markdown("### Annual Returns")
            tab1, tab2 = st.tabs(["Portfolio Returns", "Returns by Ticker"])

            with tab1:
                # Get data from session state
                close_prices = st.session_state.get('close_prices')
                weights = st.session_state.get('portfolio_weights')
                if close_prices is not None and weights is not None:
                    # Calculate annual returns
                    annual_returns = calculate_annual_returns(close_prices, weights)
                    # Filter for selected years
                    start_year = st.session_state['start_year']
                    end_year = st.session_state['end_year']
                    annual_returns = annual_returns[
                        (annual_returns.index.year >= start_year) & 
                        (annual_returns.index.year <= end_year)
                    ]
                    if not annual_returns.empty:
                        # Create bar chart with color coding
                        fig_annual = go.Figure()
                        for year, ret in annual_returns.items():
                            color = 'green' if ret > 0 else 'red'
                            fig_annual.add_trace(go.Bar(
                                x=[year.year],
                                y=[ret],
                                name=str(year.year),
                                marker_color=color,
                                text=[f"{ret:.1%}"],
                                textposition='auto'
                            ))
                        fig_annual.update_layout(
                            title="Portfolio Annual Returns",
                            xaxis_title="Year",
                            yaxis_title="Return",
                            yaxis_tickformat=".0%",
                            showlegend=False,
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=400
                        )
                        st.plotly_chart(fig_annual, use_container_width=True)
                    else:
                        st.warning("Not enough data to calculate annual returns for selected period")
                else:
                    st.warning("Portfolio data not available. Please create a portfolio first.")

            with tab2:
                close_prices = st.session_state.get('close_prices')
                if close_prices is not None:
                    yearly_prices = close_prices.resample('Y').last()
                    annual_data = []

                    for ticker in close_prices.columns:
                        returns = yearly_prices[ticker].pct_change().dropna()
                        for dt, value in returns.items():
                            if st.session_state['start_year'] <= dt.year <= st.session_state['end_year']:
                                annual_data.append({
                                    "Year": dt.year,
                                    "Ticker": ticker,
                                    "Return": value
                                })

                    df = pd.DataFrame(annual_data)
                    if not df.empty:
                        fig = px.bar(
                            df, 
                            x="Year", 
                            y="Return", 
                            color="Ticker", 
                            barmode="group",
                            text=df["Return"].map(lambda x: f"{x:.1%}"),
                            title="Annual Returns by Ticker",
                            color_discrete_sequence=['#1f77b4', '#2ca9e1', '#4dabf7', '#74c0fc', '#90d8fd', 
                                                   '#a5d8ff', '#c4e0ff', '#d6e9ff', '#63be7b', '#8fd694']
                        )
                        fig.update_layout(
                            yaxis_tickformat=".0%",
                            height=450,
                            margin=dict(l=20, r=20, t=40, b=20),
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="center",
                                x=0.5
                            )
                        )
                        fig.update_traces(
                            textposition='outside',
                            textfont_size=10,
                            marker_line_color='rgba(0,0,0,0.5)',
                            marker_line_width=1
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough data to calculate per-ticker annual returns.")
                else:
                    st.warning("Portfolio data not available. Please create a portfolio first.")

            # --- Efficient Frontier Visualization ---
            st.markdown("### Efficient Frontier Portfolios")
            # Get data from session state
            close_prices = st.session_state.get('close_prices')
            weights = st.session_state.get('portfolio_weights')
            selected_tickers = st.session_state.get('selected_tickers')
            
            if close_prices is not None and weights is not None and selected_tickers:
                with st.spinner("Calculating Efficient Frontier..."):
                    # Calculate returns and covariance matrix
                    returns = close_prices.pct_change().dropna()
                    mean_returns = returns.mean() * 252
                    cov_matrix = returns.cov() * 252
                    
                    # Helper functions
                    def portfolio_return(weights, mean_returns):
                        return np.dot(weights, mean_returns)
                        
                    def portfolio_volatility(weights, mean_returns, cov_matrix):
                        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                        
                    def get_ef_portfolios(mean_returns, cov_matrix, risk_free_rate=0.0, num_portfolios=20):
                        args = (mean_returns, cov_matrix)
                        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                        bounds = tuple((0,1) for _ in range(len(mean_returns)))
                        
                        # Minimum volatility portfolio
                        min_vol = minimize(portfolio_volatility, 
                                          len(mean_returns)*[1./len(mean_returns),], 
                                          args=args, 
                                          method='SLSQP', 
                                          bounds=bounds, 
                                          constraints=constraints)
                        
                        # Efficient frontier
                        target_returns = np.linspace(min_vol['fun'], max(mean_returns), num_portfolios)
                        efficient_portfolios = []
                        
                        for ret in target_returns:
                            constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x, mean_returns) - ret},
                                          {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                            opt = minimize(portfolio_volatility, 
                                          len(mean_returns)*[1./len(mean_returns),], 
                                          args=args, 
                                          method='SLSQP', 
                                          bounds=bounds, 
                                          constraints=constraints)
                            efficient_portfolios.append({
                                'weights': opt.x,
                                'return': ret,
                                'volatility': opt.fun,
                                'sharpe': (ret - risk_free_rate) / opt.fun if opt.fun > 0 else 0
                            })
                        return efficient_portfolios
                        
                    # Get efficient frontier portfolios
                    ef_portfolios = get_ef_portfolios(mean_returns, cov_matrix, num_portfolios=20)
                    st.session_state['ef_portfolios'] = ef_portfolios  # Store for click events
                    
                    # Create table of portfolios
                    portfolio_table = []
                    for i, pf in enumerate(ef_portfolios):
                        row = {
                            '#': i+1,
                            **{ticker: f"{pf['weights'][j]*100:.2f}%" for j, ticker in enumerate(selected_tickers)},
                            'Expected Return': f"{pf['return']:.2%}",
                            'Standard Deviation': f"{pf['volatility']:.2%}",
                            'Sharpe Ratio': f"{pf['sharpe']:.3f}"
                        }
                        portfolio_table.append(row)
                    
                    # Find max Sharpe portfolio
                    max_sharpe_portfolio = max(ef_portfolios, key=lambda x: x['sharpe'])
                    max_sharpe_idx = next(i for i, pf in enumerate(ef_portfolios) if pf['sharpe'] == max_sharpe_portfolio['sharpe'])
                    
                    # Apply styling to highlight max Sharpe row
                    def highlight_max_sharpe(row):
                        if row['#'] == max_sharpe_idx + 1:  # +1 because we start numbering from 1
                            return ['background-color: #63be7b; color: white; font-weight: bold'] * len(row)
                        return [''] * len(row)
                    
                    # Display the table
                    st.dataframe(
                        pd.DataFrame(portfolio_table).style.apply(highlight_max_sharpe, axis=1),
                        use_container_width=True
                    )
                    st.caption(f"*Row {max_sharpe_idx + 1} is the optimal portfolio (max Sharpe Ratio)*")
                    
                    # Create scatter plot
                    fig = go.Figure()
                    
                    # Add efficient frontier line
                    fig.add_trace(go.Scatter(
                        x=[pf['volatility'] for pf in ef_portfolios],
                        y=[pf['return'] for pf in ef_portfolios],
                        mode='lines+markers',
                        name='Efficient Frontier',
                        line=dict(color='royalblue', width=2),
                        marker=dict(size=8, color='royalblue'),
                        customdata=np.arange(len(ef_portfolios)),  # Store portfolio index
                        hovertemplate="<b>Return:</b> %{y:.2%}<br>" +
                                    "<b>Volatility:</b> %{x:.2%}<br>" +
                                    "<extra></extra>"
                    ))
                    
                    # Highlight current portfolio
                    current_vol = portfolio_volatility(weights, mean_returns, cov_matrix)
                    current_ret = portfolio_return(weights, mean_returns)
                    fig.add_trace(go.Scatter(
                        x=[current_vol],
                        y=[current_ret],
                        mode='markers',
                        name='Your Portfolio',
                        marker=dict(size=12, color='red', symbol='star'),
                        hovertemplate="<b>Your Portfolio</b><br>" +
                                    "<b>Return:</b> %{y:.2%}<br>" +
                                    "<b>Volatility:</b> %{x:.2%}<br>" +
                                    "<extra></extra>"
                    ))
                    
                    # Highlight max Sharpe portfolio
                    fig.add_trace(go.Scatter(
                        x=[max_sharpe_portfolio['volatility']],
                        y=[max_sharpe_portfolio['return']],
                        mode='markers',
                        name='Max Sharpe Ratio',
                        marker=dict(size=12, color='green', symbol='diamond'),
                        hovertemplate="<b>Max Sharpe</b><br>" +
                                    "<b>Return:</b> %{y:.2%}<br>" +
                                    "<b>Volatility:</b> %{x:.2%}<br>" +
                                    "<extra></extra>"
                    ))
                    
                    fig.update_layout(
                        title="Efficient Frontier",
                        xaxis_title="Volatility (Standard Deviation)",
                        yaxis_title="Expected Return",
                        yaxis_tickformat=".0%",
                        xaxis_tickformat=".0%",
                        hovermode="closest",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

            # --- Improved Toggle Buttons and Controls ---
            st.markdown("---")
            st.markdown("### Dashboard Controls")

            # Add a session state key for Monte Carlo toggle if not present
            if 'show_monte_carlo' not in st.session_state:
                st.session_state['show_monte_carlo'] = False

            toggle_cols = st.columns([1, 1, 1, 1])  # Now four columns for four buttons

            with toggle_cols[0]:
                if st.button(
                    "📊 Show Charts" if not st.session_state.get('show_charts') else "📊 Hide Charts",
                    use_container_width=True,
                    type="primary" if st.session_state.get('show_charts') else "secondary",
                    key="charts_toggle_button"
                ):
                    st.session_state['show_charts'] = not st.session_state.get('show_charts')
                    st.session_state['show_financials'] = False
                    st.session_state['show_monte_carlo'] = False
                    st.session_state['show_market_overview'] = False
                    st.rerun()

            with toggle_cols[1]:
                if st.button(
                    "💰 Show Financials" if not st.session_state.get('show_financials') else "💰 Hide Financials",
                    use_container_width=True,
                    type="primary" if st.session_state.get('show_financials') else "secondary",
                    key="financials_toggle_button"
                ):
                    st.session_state['show_financials'] = not st.session_state.get('show_financials')
                    st.session_state['show_charts'] = False
                    st.session_state['show_monte_carlo'] = False
                    st.session_state['show_market_overview'] = False
                    st.rerun()

            with toggle_cols[2]:
                if st.button(
                    "📉 Monte Carlo Simulation" if not st.session_state.get('show_monte_carlo') else "📉 Hide Monte Carlo",
                    use_container_width=True,
                    type="primary" if st.session_state.get('show_monte_carlo') else "secondary",
                    key="monte_carlo_toggle_button"
                ):
                    st.session_state['show_monte_carlo'] = not st.session_state.get('show_monte_carlo')
                    st.session_state['show_charts'] = False
                    st.session_state['show_financials'] = False
                    st.session_state['show_market_overview'] = False
                    st.rerun()

            with toggle_cols[3]:
                if st.button(
                    "🌎 Market Overview" if not st.session_state.get('show_market_overview') else "🌎 Hide Market",
                    use_container_width=True,
                    type="primary" if st.session_state.get('show_market_overview') else "secondary",
                    key="market_toggle_button"
                ):
                    st.session_state['show_market_overview'] = not st.session_state.get('show_market_overview')
                    st.session_state['show_charts'] = False
                    st.session_state['show_financials'] = False
                    st.session_state['show_monte_carlo'] = False
                    st.rerun()

            # --- Monte Carlo Simulation Section (now toggled) ---
            if st.session_state.get('show_monte_carlo') and st.session_state.get('optimal_metrics'):
                st.markdown("### 📉 Monte Carlo Portfolio Simulation")
                # Show selected tickers and weights
                alloc = st.session_state.get('optimal_result', {})
                if alloc:
                    alloc_str = ", ".join([f"{ticker}: {weight:.2f}%" for ticker, weight in alloc.items()])
                    st.info(f"**Portfolio:** {alloc_str}")
                expected_return = st.session_state['optimal_metrics'].get('expected_return', None)
                volatility = st.session_state['optimal_metrics'].get('volatility', None)
                if expected_return is not None and volatility is not None:
                    st.info(f"Expected Return: **{expected_return:.2%}**, Volatility: **{volatility:.2%}**")
                else:
                    st.warning("Please create a portfolio to enable simulation.")
                start_value = st.number_input("Starting Portfolio Value ($)", value=10000)
                years = st.slider("Simulation Time Horizon (Years)", 1, 30, 10)
                simulations = st.slider("Number of Simulations", 100, 2000, 500, step=100)
                if st.button("Run Monte Carlo Simulation"):
                    if expected_return is None or volatility is None:
                        st.error("Portfolio metrics not found. Please create a portfolio first.")
                    else:
                        results = monte_carlo_simulation(start_value, expected_return, volatility, years, simulations)
                        final_values = results[:, -1]
                        fig = go.Figure()
                        for i in range(min(50, simulations)):
                            fig.add_trace(go.Scatter(y=results[i], mode='lines', line=dict(width=1), showlegend=False))
                        fig.update_layout(
                            title="Monte Carlo Simulation of Portfolio Value",
                            xaxis_title="Days",
                            yaxis_title="Portfolio Value ($)",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        def pct_str(final):
                            pct = (final - start_value) / start_value * 100
                            return f"({pct:+.2f}%)"
                        median_val = np.median(final_values)
                        p5_val = np.percentile(final_values, 5)
                        p95_val = np.percentile(final_values, 95)
                        st.write(f"**Median Final Value:** ${median_val:,.2f} {pct_str(median_val)}")
                        st.write(f"**5th Percentile (Worst-Case):** ${p5_val:,.2f} {pct_str(p5_val)}")
                        st.write(f"**95th Percentile (Best-Case):** ${p95_val:,.2f} {pct_str(p95_val)}")

# --- Dashboard Sections ---
if st.session_state.get('portfolio_created'):
    # --- Charts Section ---
    if st.session_state.get('show_charts') and not st.session_state.get('show_financials') and not st.session_state.get('show_monte_carlo') and not st.session_state.get('show_market_overview'):
        st.markdown("### Price Charts and Key Metrics")
        
        # Period selector with more options
        period_options = {
            "1D": ("1d", "15m"),  # Changed interval from 5m to 15m for 1D
            "5D": ("5d", "5m"),   # Changed interval from 15m to 5m for 5D
            "1M": ("1mo", "30m"),
            "6M": ("6mo", "1d"),
            "YTD": ("ytd", "1d"),
            "1Y": ("1y", "1d"),
            "5Y": ("5y", "1wk"),
            "10Y": ("10y", "1mo"),
            "MAX": ("max", "1mo")
        }
        period_label = st.selectbox("Select Time Period", list(period_options.keys()))
        period, interval = period_options[period_label]

        if st.session_state.get('selected_tickers'):
            # Add tabs for each ticker
            tabs = st.tabs([f"📈 {ticker}" for ticker in st.session_state['selected_tickers']])
            
            for i, ticker in enumerate(st.session_state['selected_tickers']):
                with tabs[i]:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Enhanced chart with moving averages (removed volume)
                        try:
                            hist = yf.download(ticker, period=period, interval=interval)
                            if hist is not None and not hist.empty:
                                hist = hist.copy()
                                hist.index = pd.to_datetime(hist.index)
                                
                                # Get close price series
                                if isinstance(hist.columns, pd.MultiIndex):
                                    close_series = hist[('Close', ticker)].dropna() if ('Close', ticker) in hist.columns else None
                                else:
                                    close_series = hist['Close'].dropna() if 'Close' in hist.columns else None
                                
                                if close_series is not None and not close_series.empty:
                                    fig = go.Figure()
                                    
                                    # Main price line - thicker and more prominent
                                    fig.add_trace(go.Scatter(
                                        x=close_series.index,
                                        y=close_series,
                                        mode='lines',
                                        name='Price',
                                        line=dict(color='#1f77b4', width=3)  # Thicker line
                                    ))
                                    
                                    # Layout configuration - cleaner and more compact
                                    fig.update_layout(
                                        title=f"{ticker} Price ({period_label})",
                                        xaxis_title=None,  # Remove x-axis title
                                        yaxis_title=None,  # Remove y-axis title
                                        margin=dict(l=20, r=20, t=40, b=20),
                                        height=350,  # Slightly shorter
                                        showlegend=False,  # Remove legend
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        xaxis=dict(
                                            showgrid=True,
                                            gridcolor='rgba(211, 211, 211, 0.3)',  # Lighter grid
                                            showline=True,
                                            linecolor='rgba(211, 211, 211, 0.5)'
                                        ),
                                        yaxis=dict(
                                            showgrid=True,
                                            gridcolor='rgba(211, 211, 211, 0.3)',  # Lighter grid
                                            showline=True,
                                            linecolor='rgba(211, 211, 211, 0.5)',
                                            side='right'  # Move y-axis to right
                                        )
                                    )
                                    
                                    # Format x-axis based on period
                                    if period_label == "1D":
                                        fig.update_xaxes(tickformat="%H:%M")
                                    else:
                                        fig.update_xaxes(tickformat="%b %d")
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("No price data available for this ticker and period after cleaning.")
                        except Exception as e:
                            st.error(f"Error loading chart data for {ticker}: {str(e)}")
                    
                    with col2:
                        # Enhanced key metrics in a compact card layout
                        st.markdown(f"### {ticker} Metrics")
                        
                        try:
                            yahoo_ticker = to_yahoo_ticker(ticker)
                            info = yf.Ticker(yahoo_ticker).info
                            
                            if not info or info == {}:
                                st.warning(f"No summary info found for {ticker}")
                            else:
                                # Get historical prices
                                historical_prices = get_historical_closes(ticker)
                                
                                if historical_prices:
                                    # Current price
                                    current_price = historical_prices["current"]
                                    price_display = f"${current_price:,.2f}" if isinstance(current_price, (int, float)) else 'N/A'
                                    
                                    # Create the price display card
                                    st.markdown(f"""
                                    <div style="background-color: #1f77b4; border-radius: 10px; padding: 15px; margin-bottom: 15px; color: white;">
                                        <div style="font-size: 1.8rem; font-weight: bold; margin-bottom: 5px;">{price_display}</div>
                                        <div style="font-size: 1rem; margin: 5px 0;">Latest Close Price</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Historical prices table
                                    st.markdown("#### Historical Close Prices")
                                    
                                    # Create a dataframe for the historical prices
                                    hist_df = pd.DataFrame({
                                        "Period": ["1 Month Ago", "6 Months Ago", "YTD Start", "1 Year Ago", "5 Years Ago"],
                                        "Price": [
                                            f"${historical_prices['1m']:,.2f}" if historical_prices['1m'] else 'N/A',
                                            f"${historical_prices['6m']:,.2f}" if historical_prices['6m'] else 'N/A',
                                            f"${historical_prices['ytd']:,.2f}" if historical_prices['ytd'] else 'N/A',
                                            f"${historical_prices['1y']:,.2f}" if historical_prices['1y'] else 'N/A',
                                            f"${historical_prices['5y']:,.2f}" if historical_prices['5y'] else 'N/A'
                                        ],
                                        "% Change": [
                                            f"{((current_price - historical_prices['1m']) / historical_prices['1m'] * 100):+.2f}%" if historical_prices['1m'] else 'N/A',
                                            f"{((current_price - historical_prices['6m']) / historical_prices['6m'] * 100):+.2f}%" if historical_prices['6m'] else 'N/A',
                                            f"{((current_price - historical_prices['ytd']) / historical_prices['ytd'] * 100):+.2f}%" if historical_prices['ytd'] else 'N/A',
                                            f"{((current_price - historical_prices['1y']) / historical_prices['1y'] * 100):+.2f}%" if historical_prices['1y'] else 'N/A',
                                            f"{((current_price - historical_prices['5y']) / historical_prices['5y'] * 100):+.2f}%" if historical_prices['5y'] else 'N/A'
                                        ]
                                    })
                                    
                                    # Style the % Change column with color coding
                                    def color_pct_change(val):
                                        if val == 'N/A':
                                            return 'color: black'
                                        try:
                                            pct = float(val.replace('%', ''))
                                            return 'color: green' if pct > 0 else 'color: red'
                                        except:
                                            return 'color: black'
                                    
                                    styled_df = hist_df.style.applymap(color_pct_change, subset=['% Change'])
                                    
                                    # Display the table with minimal styling
                                    st.dataframe(
                                        styled_df,
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            "Period": st.column_config.TextColumn("Period", width="medium"),
                                            "Price": st.column_config.TextColumn("Price", width="small"),
                                            "% Change": st.column_config.TextColumn("% Change", width="small")
                                        }
                                    )
                                
                                # Key metrics in a table-like format
                                metrics = [
                                    ("Market Cap", format_market_cap(info.get('marketCap', 'N/A'))),
                                    ("PE Ratio", f"{info.get('trailingPE', 'N/A'):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else 'N/A'),
                                    ("Beta", f"{info.get('beta', 'N/A'):.2f}" if isinstance(info.get('beta'), (int, float)) else 'N/A'),
                                    ("52W High", f"${info.get('fiftyTwoWeekHigh', 'N/A'):,.2f}" if isinstance(info.get('fiftyTwoWeekHigh'), (int, float)) else 'N/A'),
                                    ("52W Low", f"${info.get('fiftyTwoWeekLow', 'N/A'):,.2f}" if isinstance(info.get('fiftyTwoWeekLow'), (int, float)) else 'N/A')
                                ]
                                
                                st.markdown("""
                                <style>
                                    .metric-row {
                                        display: flex;
                                        justify-content: space-between;
                                        padding: 8px 0;
                                        border-bottom: 1px solid #eee;
                                    }
                                    .metric-label {
                                        font-weight: bold;
                                        color: #555;
                                    }
                                    .metric-value {
                                        text-align: right;
                                    }
                                </style>
                                """, unsafe_allow_html=True)
                                
                                for label, value in metrics:
                                    st.markdown(f"""
                                    <div class="metric-row">
                                        <span class="metric-label">{label}</span>
                                        <span class="metric-value">{value}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Company info section
                                sector = info.get('sector', 'N/A')
                                industry = info.get('industry', 'N/A')
                                employees = info.get('fullTimeEmployees', 'N/A')
                                
                                # Fallback to Kaggle CSV if missing
                                if kaggle_df is not None:
                                    ticker_row = kaggle_df[kaggle_df['Symbol'].str.upper() == ticker.upper()]
                                    if not ticker_row.empty:
                                        if sector == 'N/A':
                                            sector = ticker_row['Sector'].values[0] if 'Sector' in ticker_row else 'N/A'
                                        if industry == 'N/A':
                                            industry = ticker_row['Industry'].values[0] if 'Industry' in ticker_row else 'N/A'
                                        if employees == 'N/A' and 'Full Time Employees' in ticker_row:
                                            employees = ticker_row['Full Time Employees'].values[0]
                                
                                st.markdown(f"""
                                <div style="margin-top: 20px; font-size: 0.9rem;">
                                    <div><strong>Sector:</strong> {sector}</div>
                                    <div><strong>Industry:</strong> {industry}</div>
                                    <div><strong>Employees:</strong> {employees if employees != 'N/A' else 'N/A'}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.error(f"Error fetching metrics for {ticker}: {str(e)}")

    # --- Financials Section ---
    if st.session_state.get('show_financials') and not st.session_state.get('show_charts') and not st.session_state.get('show_monte_carlo') and not st.session_state.get('show_market_overview'):
        st.markdown("### Financial Analysis")
        
        selected_ticker = st.selectbox(
            "Select Company to View Financials",
            options=st.session_state.get('selected_tickers', []),
            key="financial_ticker_selector"
        )

        if selected_ticker:
            try:
                # When a company is selected, preload all financial data and build year dictionaries before defining tabs
                income_data = get_financial_data(selected_ticker, 'income')
                balance_data = get_financial_data(selected_ticker, 'balance')
                cashflow_data = get_financial_data(selected_ticker, 'cashflow')
                # Helper to normalize index
                def normalize_index(df):
                    df = df.copy()
                    df.index = [str(idx).lower() for idx in df.index]
                    return df
                # Helper to extract rows for a metric map
                def extract_rows(data, metric_map, year_cols):
                    rows = []
                    for display_name, aliases in metric_map:
                        found = False
                        for alias in aliases:
                            if alias in data.index:
                                row = data.loc[alias, year_cols]
                                found = True
                                break
                        if not found:
                            row = [None] * len(year_cols)
                        rows.append([display_name] + list(row))
                    return rows
                # Define year_cols for each statement
                def get_year_cols(data):
                    if data is None or data.empty:
                        return []
                    current_year = datetime.now().year
                    years = list(range(current_year-1, current_year-5, -1))
                    year_cols = []
                    for col in data.columns:
                        if isinstance(col, (pd.Timestamp, datetime)):
                            if col.year in years:
                                year_cols.append(col)
                        elif str(col).isdigit() and int(col) in years:
                            year_cols.append(col)
                    year_cols.sort(reverse=True)
                    return year_cols
                # Metric maps
                income_metric_map = [
                    ("Total Revenue", ["total revenue", "totalrevenue"]),
                    ("Cost of Revenue", ["cost of revenue", "cost of revenue", "costofrevenue"]),
                    ("Gross Profit", ["gross profit", "grossprofit"]),
                    ("Operating Income", ["operating income", "operatingincome"]),
                    ("EBIT", ["ebit", "earnings before interest and taxes"]),
                    ("EBITDA", ["ebitda"]),
                    ("Total Expenses", ["total expenses", "totalexpenses"]),
                    ("Diluted EPS", ["diluted eps", "dilutedeps"]),
                    ("Net Income Common Stockholders", ["net income common stockholders", "net income", "netincome"])
                ]
                balance_metric_map = [
                    ("Total Assets", ["total assets", "totalassets"]),
                    ("Current Assets", ["current assets"]),
                    ("Total Liabilities", ["total liabilities", "totalliabilities", "total liabilities net minority interest"]),
                    ("Current Liabilities", ["current liabilities"]),
                    ("Total Equity", ["total equity", "total stockholder equity", "totalequity", "totalstockholderequity", "total equity gross minority interest"]),
                    ("Total Capitalization", ["total capitalization", "totalcapitalization"]),
                    ("Net Tangible Assets", ["net tangible assets", "nettangibleassets"]),
                    ("Working Capital", ["working capital", "workingcapital"]),
                    ("Invested Capital", ["invested capital", "investedcapital"]),
                    ("Total Debt", ["total debt", "totaldebt"]),
                    ("Inventory", ["inventory"]),
                    ("Net Receivables", ["net receivables", "accounts receivable", "accountsreceivable"])
                ]
                cashflow_metric_map = [
                    ("Operating Cash Flow", ["operating cash flow", "total cash from operating activities", "net cash provided by operating activities"]),
                    ("Investing Cash Flow", ["investing cash flow", "total cashflows from investing activities", "net cash used for investing activities"]),
                    ("Financing Cash Flow", ["financing cash flow", "total cash from financing activities", "net cash provided by financing activities"]),
                    ("End Cash Flow", ["end cash position", "cash at end of period", "cash and cash equivalents at end of year"]),
                    ("Capital Expenditure", ["capital expenditure", "capital expenditures"]),
                    ("Free Cash Flow", ["free cash flow"])
                ]
                # Normalize and extract
                income_dict_by_year = {}
                balance_dict_by_year = {}
                cashflow_dict_by_year = {}
                if income_data is not None and not income_data.empty:
                    income_data = normalize_index(income_data)
                    year_cols = get_year_cols(income_data)
                    income_rows = extract_rows(income_data, income_metric_map, year_cols)
                    for i, col in enumerate(year_cols):
                        year = str(col.year) if hasattr(col, 'year') else str(col)
                        income_dict_by_year[year] = {r[0]: r[i+1] for r in income_rows}
                if balance_data is not None and not balance_data.empty:
                    balance_data = normalize_index(balance_data)
                    year_cols = get_year_cols(balance_data)
                    balance_rows = extract_rows(balance_data, balance_metric_map, year_cols)
                    for i, col in enumerate(year_cols):
                        year = str(col.year) if hasattr(col, 'year') else str(col)
                        balance_dict_by_year[year] = {r[0]: r[i+1] for r in balance_rows}
                if cashflow_data is not None and not cashflow_data.empty:
                    cashflow_data = normalize_index(cashflow_data)
                    year_cols = get_year_cols(cashflow_data)
                    cashflow_rows = extract_rows(cashflow_data, cashflow_metric_map, year_cols)
                    for i, col in enumerate(year_cols):
                        year = str(col.year) if hasattr(col, 'year') else str(col)
                        cashflow_dict_by_year[year] = {r[0]: r[i+1] for r in cashflow_rows}
                # Now define the tabs
                fin_tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow", "Key Ratios"])

                with fin_tabs[0]:
                    st.subheader(f"{selected_ticker} Income Statement")
                    if income_rows:
                        formatted_data = pd.DataFrame(
                            income_rows,
                            columns=["Metric"] + [str(col.year) if hasattr(col, 'year') else str(col) for col in year_cols]
                        )
                        # Format values: currency for all except EPS (2 decimals)
                        for idx, row in enumerate(formatted_data["Metric"]):
                            if row == "Diluted EPS":
                                formatted_data.iloc[idx, 1:] = formatted_data.iloc[idx, 1:].apply(lambda x: f"{x:,.2f}" if pd.notna(x) and x != None else 'N/A')
                            else:
                                formatted_data.iloc[idx, 1:] = formatted_data.iloc[idx, 1:].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and x != None else 'N/A')
                        st.dataframe(formatted_data, use_container_width=True)
                        # --- Horizontal Analysis ---
                        st.markdown("#### Horizontal Analysis (Year-over-Year % Change)")
                        numeric_df = pd.DataFrame(income_rows, columns=["Metric"] + [str(col.year) if hasattr(col, 'year') else str(col) for col in year_cols])
                        numeric_df = numeric_df.set_index("Metric")
                        numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
                        horiz_years = [str(y) for y in sorted([int(c) for c in numeric_df.columns], reverse=True)[:3]]
                        horiz_df = pd.DataFrame(index=numeric_df.index)
                        for i, year in enumerate(horiz_years):
                            prev_idx = numeric_df.columns.get_loc(year) + 1
                            if prev_idx < len(numeric_df.columns):
                                prev_year = numeric_df.columns[prev_idx]
                                horiz_df[year] = [horizontal_analysis(numeric_df.loc[metric, prev_year], numeric_df.loc[metric, year]) for metric in numeric_df.index]
                            else:
                                horiz_df[year] = None
                        horiz_df = horiz_df[horiz_years]
                        horiz_df = horiz_df.applymap(lambda x: f"<span style='color:green'>{x:.2f}%</span>" if x is not None and x > 0 else (f"<span style='color:red'>{x:.2f}%</span>" if x is not None and x < 0 else 'N/A'))
                        horiz_df = horiz_df.reset_index()
                        st.write(horiz_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                        # --- Vertical Analysis ---
                        st.markdown("#### Vertical Analysis (as % of Total Revenue)")
                        if "Total Revenue" in numeric_df.index:
                            total_revenue = numeric_df.loc["Total Revenue"]
                            vert_df = numeric_df.divide(total_revenue, axis=1)
                            vert_df = vert_df.applymap(lambda x: f"{x:.2%}" if pd.notna(x) else 'N/A')
                            vert_df = vert_df.reset_index()
                            st.dataframe(vert_df, use_container_width=True)
                        else:
                            st.warning("Could not find 'Total Revenue' for vertical analysis.")

                with fin_tabs[1]:
                    st.subheader(f"{selected_ticker} Balance Sheet")
                    if balance_rows:
                        formatted_data = pd.DataFrame(
                            balance_rows,
                            columns=["Metric"] + [str(col.year) if hasattr(col, 'year') else str(col) for col in year_cols]
                        )
                        for idx, row in enumerate(formatted_data["Metric"]):
                            formatted_data.iloc[idx, 1:] = formatted_data.iloc[idx, 1:].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and x != None else 'N/A')
                        st.dataframe(formatted_data, use_container_width=True)
                        # --- Horizontal Analysis ---
                        st.markdown("#### Horizontal Analysis (Year-over-Year % Change)")
                        numeric_df = pd.DataFrame(balance_rows, columns=["Metric"] + [str(col.year) if hasattr(col, 'year') else str(col) for col in year_cols])
                        numeric_df = numeric_df.set_index("Metric")
                        numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
                        horiz_years = [str(y) for y in sorted([int(c) for c in numeric_df.columns], reverse=True)[:3]]
                        horiz_df = pd.DataFrame(index=numeric_df.index)
                        for i, year in enumerate(horiz_years):
                            prev_idx = numeric_df.columns.get_loc(year) + 1
                            if prev_idx < len(numeric_df.columns):
                                prev_year = numeric_df.columns[prev_idx]
                                horiz_df[year] = [horizontal_analysis(numeric_df.loc[metric, prev_year], numeric_df.loc[metric, year]) for metric in numeric_df.index]
                            else:
                                horiz_df[year] = None
                        horiz_df = horiz_df[horiz_years]
                        horiz_df = horiz_df.applymap(lambda x: f"<span style='color:green'>{x:.2f}%</span>" if x is not None and x > 0 else (f"<span style='color:red'>{x:.2f}%</span>" if x is not None and x < 0 else 'N/A'))
                        horiz_df = horiz_df.reset_index()
                        st.write(horiz_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                        # --- Vertical Analysis ---
                        st.markdown("#### Vertical Analysis (as % of Total Assets)")
                        if "Total Assets" in numeric_df.index:
                            total_assets = numeric_df.loc["Total Assets"]
                            vert_df = numeric_df.divide(total_assets, axis=1)
                            vert_df = vert_df.applymap(lambda x: f"{x:.2%}" if pd.notna(x) else 'N/A')
                            vert_df = vert_df.reset_index()
                            st.dataframe(vert_df, use_container_width=True)
                        else:
                            st.warning("Could not find 'Total Assets' for vertical analysis.")

                with fin_tabs[2]:
                    st.subheader(f"{selected_ticker} Cash Flow")
                    if cashflow_rows:
                        formatted_data = pd.DataFrame(
                            cashflow_rows,
                            columns=["Metric"] + [str(col.year) if hasattr(col, 'year') else str(col) for col in year_cols]
                        )
                        for idx, row in enumerate(formatted_data["Metric"]):
                            formatted_data.iloc[idx, 1:] = formatted_data.iloc[idx, 1:].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and x != None else 'N/A')
                        st.dataframe(formatted_data, use_container_width=True)
                        # --- Horizontal Analysis ---
                        st.markdown("#### Horizontal Analysis (Year-over-Year % Change)")
                        numeric_df = pd.DataFrame(cashflow_rows, columns=["Metric"] + [str(col.year) if hasattr(col, 'year') else str(col) for col in year_cols])
                        numeric_df = numeric_df.set_index("Metric")
                        numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
                        horiz_years = [str(y) for y in sorted([int(c) for c in numeric_df.columns], reverse=True)[:3]]
                        horiz_df = pd.DataFrame(index=numeric_df.index)
                        for i, year in enumerate(horiz_years):
                            prev_idx = numeric_df.columns.get_loc(year) + 1
                            if prev_idx < len(numeric_df.columns):
                                prev_year = numeric_df.columns[prev_idx]
                                horiz_df[year] = [horizontal_analysis(numeric_df.loc[metric, prev_year], numeric_df.loc[metric, year]) for metric in numeric_df.index]
                            else:
                                horiz_df[year] = None
                        horiz_df = horiz_df[horiz_years]
                        horiz_df = horiz_df.applymap(lambda x: f"<span style='color:green'>{x:.2f}%</span>" if x is not None and x > 0 else (f"<span style='color:red'>{x:.2f}%</span>" if x is not None and x < 0 else 'N/A'))
                        horiz_df = horiz_df.reset_index()
                        st.write(horiz_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                        # --- Vertical Analysis ---
                        st.markdown("#### Vertical Analysis (as % of Operating Cash Flow)")
                        op_cf_metric = None
                        for op_cf_name in ["Operating Cash Flow", "Total Cash From Operating Activities", "Net Cash Provided By Operating Activities"]:
                            if op_cf_name in numeric_df.index:
                                op_cf_metric = numeric_df.loc[op_cf_name]
                                break
                        if op_cf_metric is not None:
                            vert_df = numeric_df.divide(op_cf_metric, axis=1)
                            vert_df = vert_df.applymap(lambda x: f"{x:.2%}" if pd.notna(x) else 'N/A')
                            vert_df = vert_df.reset_index()
                            st.dataframe(vert_df, use_container_width=True)
                        else:
                            st.warning("Could not find 'Operating Cash Flow' for vertical analysis.")

                with fin_tabs[3]:
                    st.subheader("📐 Key Financial Ratios")
                    # Only show if we have at least one year of all statements
                    if income_dict_by_year and balance_dict_by_year:
                        ratio_matrix = {}
                        for year in year_cols:
                            year_str = str(year.year) if hasattr(year, 'year') else str(year)
                            income = income_dict_by_year.get(year_str, {})
                            balance = balance_dict_by_year.get(year_str, {})
                            cash = cashflow_dict_by_year.get(year_str, {})
                            def val(d, key):
                                try:
                                    return float(str(d.get(key, None)).replace(',', '').replace('$', ''))
                                except:
                                    return None
                            # Extract correct values
                            revenue = val(income, "Total Revenue")
                            gross_profit = val(income, "Gross Profit")
                            operating_income = val(income, "Operating Income")
                            net_income = val(income, "Net Income Common Stockholders")
                            cogs = val(income, "Cost of Revenue")
                            total_assets = val(balance, "Total Assets")
                            total_equity = val(balance, "Total Equity")
                            current_assets = val(balance, "Current Assets")
                            current_liabilities = val(balance, "Current Liabilities")
                            inventory = val(balance, "Inventory")
                            # Fallbacks if not available
                            if current_assets is None:
                                current_assets = total_assets
                            if current_liabilities is None:
                                current_liabilities = val(balance, "Total Liabilities")
                            if inventory is None:
                                inventory = val(balance, "Net Tangible Assets")
                            # For efficiency ratios, need previous year for averages
                            prev_year = str(int(year_str)-1)
                            prev_balance = balance_dict_by_year.get(prev_year, {})
                            avg_receivables = avg_inventory = avg_payables = None
                            if prev_balance:
                                ar = val(balance, "Net Receivables")
                                prev_ar = val(prev_balance, "Net Receivables")
                                if ar is not None and prev_ar is not None:
                                    avg_receivables = (ar + prev_ar) / 2
                                inv = inventory
                                prev_inv = val(prev_balance, "Inventory")
                                if inv is not None and prev_inv is not None:
                                    avg_inventory = (inv + prev_inv) / 2
                                ap = val(balance, "Total Liabilities")
                                prev_ap = val(prev_balance, "Total Liabilities")
                                if ap is not None and prev_ap is not None:
                                    avg_payables = (ap + prev_ap) / 2
                            purchases = cogs
                            op_cf = val(cash, "Operating Cash Flow")
                            # Compute ratios
                            ratios = {
                                "Current Ratio": current_ratio(current_assets, current_liabilities),
                                "Quick Ratio": quick_ratio(current_assets, inventory, current_liabilities),
                                "Working Capital": working_capital(current_assets, current_liabilities),
                                "Gross Margin": gross_margin(gross_profit, revenue),
                                "Operating Margin": operating_margin(operating_income, revenue),
                                "Net Margin": net_margin(net_income, revenue),
                                "ROA": return_on_assets(net_income, total_assets),
                                "ROE": return_on_equity(net_income, total_equity),
                                "DSO": dso(avg_receivables, revenue),
                                "DPO": dpo(avg_payables, purchases),
                                "DIO": dio(avg_inventory, cogs),
                                "Cash Conversion Cycle": cash_conversion_cycle(
                                    dso(avg_receivables, revenue),
                                    dio(avg_inventory, cogs),
                                    dpo(avg_payables, purchases)
                                )
                            }
                            for k, v in ratios.items():
                                if k not in ratio_matrix:
                                    ratio_matrix[k] = {}
                                if v is None:
                                    display = 'N/A'
                                elif 'Margin' in k or k in ["ROA", "ROE"]:
                                    display = f"{v:.2%}"
                                elif 'Working Capital' in k:
                                    display = f"${v:,.0f}"
                                else:
                                    display = f"{v:.2f}"
                                ratio_matrix[k][year_str] = display
                        df = pd.DataFrame(ratio_matrix).T
                        st.dataframe(df, use_container_width=True)
                        
                        # Add ratio insights
                        st.markdown("### 🧠 Ratio Insights")
                        
                        # Parse numeric values from display strings
                        parsed_ratios = {}
                        for ratio_name, year_values in ratio_matrix.items():
                            # Get the most recent year's value
                            latest_year = max(year_values.keys())
                            value = year_values[latest_year]
                            try:
                                if isinstance(value, str):
                                    # Remove % and $ signs and convert to float
                                    clean_value = value.replace('%', '').replace('$', '').replace(',', '')
                                    parsed = float(clean_value) / (100 if '%' in value else 1)
                                    parsed_ratios[ratio_name] = parsed
                                elif isinstance(value, (int, float)):
                                    parsed_ratios[ratio_name] = value
                            except:
                                parsed_ratios[ratio_name] = None
                        
                        insights = interpret_ratios(parsed_ratios)
                        if insights:
                            # Add legend
                            st.markdown("""
                            <div style="font-size: 14px; margin-bottom: 10px;">
                                <span style="margin-right: 10px;">🟢 Best</span>
                                <span style="margin-right: 10px;">🔵 Moderate</span>
                                <span style="margin-right: 10px;">🔴 Concerning</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display insights with markdown formatting
                            for msg in insights:
                                st.markdown(f"- {msg}", unsafe_allow_html=True)
                        else:
                            st.info("Not enough data to generate ratio insights.")
                    else:
                        st.info("Not enough data to compute ratios.")
            except Exception as e:
                st.error(f"Error fetching financial data for {selected_ticker}: {str(e)}")
                st.expander("Error details").code(traceback.format_exc())

# --- Market Overview Section ---
if st.session_state.get('show_market_overview'):
    st.markdown("## 🌎 Market Overview")
    
    # Add tooltip CSS
    st.markdown("""
    <style>
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 260px;
            background-color: #222;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 8px 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -130px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.9em;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Single timeframe selector for entire Market Overview
    market_timeframe = st.selectbox(
        "Select Timeframe",
        ["5D", "1M", "1Y", "5Y", "10Y"],
        index=0,
        key="market_timeframe"
    )
    
    # Map timeframe to yfinance parameters
    timeframe_map = {
        "5D": ("5d", "15m"),
        "1M": ("1mo", "1d"),
        "1Y": ("1y", "1wk"),
        "5Y": ("5y", "1mo"),
        "10Y": ("10y", "1mo")
    }
    
    yfinance_period, yfinance_interval = timeframe_map[market_timeframe]

    # ===== IMPROVED: Commodities & Forex =====
    st.markdown("### 🌍 Commodities, Forex & Crypto")

    # Add forex tooltip CSS
    st.markdown("""
    <style>
        .forex-tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        .forex-tooltip .forex-tooltiptext {
            visibility: hidden;
            width: 260px;
            background-color: #222;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 8px 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -130px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.9em;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .forex-tooltip:hover .forex-tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
    """, unsafe_allow_html=True)

    # Define asset groups
    commodities = {
        "Gold": "GC=F",
        "Crude Oil (WTI)": "CL=F",
        "Silver": "SI=F",
        "Natural Gas": "NG=F",
        "Brent Crude": "BZ=F",
        "Copper": "HG=F"
    }

    forex = {
        "EUR/USD ℹ️": ("EURUSD=X", "For every 1 Euro you get {rate} US Dollars"),
        "USD/JPY ℹ️": ("JPY=X", "For every 1 US Dollar you get {rate} Japanese Yen"),
        "GBP/USD ℹ️": ("GBPUSD=X", "For every 1 British Pound you get {rate} US Dollars"),
        "USD/CAD ℹ️": ("CAD=X", "For every 1 US Dollar you get {rate} Canadian Dollars"),
        "AUD/USD ℹ️": ("AUDUSD=X", "For every 1 Australian Dollar you get {rate} US Dollars"),
        "USD Index ℹ️": ("DX-Y.NYB", "Index of USD strength against basket of currencies")
    }

    crypto = {
        "Bitcoin USD": "BTC-USD",
        "Ethereum USD": "ETH-USD"
    }

    # Create tabs for different asset classes
    tab1, tab2, tab3 = st.tabs(["⛽ Commodities", "💱 Forex", "₿ Crypto"])

    with tab1:
        st.markdown("#### Commodities Performance")
        cols = st.columns(len(commodities))
        
        for i, (name, ticker) in enumerate(commodities.items()):
            with cols[i]:
                with st.spinner(f"Loading {name} data..."):
                    result = get_asset_price_change(ticker, period=yfinance_period)
                    if result:
                        current_price = result['current_price'].iloc[0] if hasattr(result['current_price'], 'iloc') else result['current_price']
                        change = result['pct_change'].iloc[0] if hasattr(result['pct_change'], 'iloc') else result['pct_change']
                        
                        # Format price based on commodity type
                        if "Oil" in name or "Gas" in name:
                            price = f"${current_price:.2f}"
                        elif name in ["Gold", "Silver"]:
                            price = f"${current_price:.2f}/oz"
                        else:
                            price = f"${current_price:.2f}"
                        
                        # Determine color
                        change_color = "red" if change < 0 else "green"
                        
                        # Create the metric with HTML/CSS
                        st.markdown(f"""
                        <div style="font-size: 1.1rem; font-weight: bold; margin-bottom: 5px;">{name}</div>
                        <div style="font-size: 1.5rem; font-weight: bold; margin: 5px 0;">{price}</div>
                        <div style="color: {change_color}; font-size: 1rem; margin: 5px 0;">{change:+.2f}%</div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ {name} data not available")
                        st.caption(f"Symbol: {ticker}")

    with tab2:
        st.markdown("#### Forex Performance")
        cols = st.columns(len(forex))
        
        for i, (name, (ticker, tooltip)) in enumerate(forex.items()):
            with cols[i]:
                with st.spinner(f"Loading {name} data..."):
                    result = get_asset_price_change(ticker, period=yfinance_period)
                    if result:
                        current_price = result['current_price'].iloc[0] if hasattr(result['current_price'], 'iloc') else result['current_price']
                        change = result['pct_change'].iloc[0] if hasattr(result['pct_change'], 'iloc') else result['pct_change']
                        
                        # Format the tooltip with current rate (max 4 decimals)
                        formatted_rate = f"{current_price:.4f}" if "Index" not in name else f"{current_price:.2f}"
                        formatted_tooltip = tooltip.format(rate=formatted_rate)
                        
                        # Special handling for USD Index
                        if "Index" in name:
                            price = f"{current_price:.2f}"
                            formatted_tooltip += "<br><br>🔴 Above 100 = USD strong<br>🟢 Below 100 = USD weak"
                        else:
                            price = f"{current_price:.4f}"
                        
                        # Create the metric with tooltip
                        st.markdown(f"""
                        <div style="font-size: 1.1rem; font-weight: bold; margin-bottom: 5px;">
                            {name.split('ℹ️')[0].strip()}
                            <span class="tooltip">ℹ️
                                <span class="tooltiptext">{formatted_tooltip}</span>
                            </span>
                        </div>
                        <div style="font-size: 1.5rem; font-weight: bold; margin: 5px 0;">{price}</div>
                        <div style="color: {'red' if change < 0 else 'green'}; font-size: 1rem; margin: 5px 0;">{change:+.2f}%</div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ {name} data not available")
                        st.caption(f"Symbol: {ticker}")

    with tab3:
        st.markdown("#### Cryptocurrency Performance")
        cols = st.columns(len(crypto))
        
        for i, (name, ticker) in enumerate(crypto.items()):
            with cols[i]:
                with st.spinner(f"Loading {name} data..."):
                    result = get_asset_price_change(ticker, period=yfinance_period)
                    if result:
                        current_price = result['current_price'].iloc[0] if hasattr(result['current_price'], 'iloc') else result['current_price']
                        change = result['pct_change'].iloc[0] if hasattr(result['pct_change'], 'iloc') else result['pct_change']
                        
                        # Improved crypto price formatting
                        if current_price >= 1000:
                            price = f"${current_price:,.2f}"
                        else:
                            price = f"${current_price:,.2f}"
                        
                        # Create the metric with HTML/CSS
                        st.markdown(f"""
                        <div style="font-size: 1.1rem; font-weight: bold; margin-bottom: 5px;">{name}</div>
                        <div style="font-size: 1.5rem; font-weight: bold; margin: 5px 0;">{price}</div>
                        <div style="color: {'red' if change < 0 else 'green'}; font-size: 1rem; margin: 5px 0;">{change:+.2f}%</div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ {name} data not available")
                        st.caption(f"Symbol: {ticker}")

    # --- Improved Market Snapshot Section ---
    st.markdown("### 📊 Market Snapshot")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        # Dynamic tooltip content based on timeframe
        timeframe_tooltips = {
            "5D": """
                <strong>S&P 500 (SPY) 5-Day Performance</strong><br>
                🟢 Bullish Week: +1.5% or more<br>
                🔴 Bearish Week: -1.5% or less<br>
                🟡 Sideways: Between thresholds
            """,
            "1M": """
                <strong>S&P 500 (SPY) Monthly Performance</strong><br>
                🟢 Strong Monthly Rally: +3% or more<br>
                🔴 Down Month: -3% or less<br>
                🟡 Neutral Month: Between thresholds
            """,
            "1Y": """
                <strong>S&P 500 (SPY) Yearly Performance</strong><br>
                🟢 Strong Year: +8% or more<br>
                🔴 Bearish Year: -8% or less<br>
                🟡 Flat Year: Between thresholds
            """,
            "5Y": """
                <strong>S&P 500 (SPY) 5-Year Performance</strong><br>
                🟢 Strong 5Y Growth: +35% or more (~6.2% CAGR)<br>
                🔴 Underperforming 5Y: +5% or less<br>
                🟡 Moderate 5Y: Between +5% and +35%
            """,
            "10Y": """
                <strong>S&P 500 (SPY) 10-Year Performance</strong><br>
                🟢 Strong Decade: +80% or more (~6% CAGR)<br>
                🔴 Weak Decade: +10% or less<br>
                🟡 Flat Decade: Between +10% and +80%
            """
        }
        
        st.markdown(f"""
        <div style="font-size: 1.1rem; font-weight: bold; margin-bottom: 5px;">
            S&P 500 (SPY)
            <span class="tooltip">ℹ️
                <span class="tooltiptext">{timeframe_tooltips.get(market_timeframe, "Performance metrics")}</span>
            </span>
        </div>
        """, unsafe_allow_html=True)
        try:
            spy_data = yf.download("SPY", period=yfinance_period, interval=yfinance_interval, progress=False)
            if not spy_data.empty and 'Close' in spy_data.columns:
                spy_close = float(spy_data['Close'].iloc[-1])
                spy_change = ((spy_close - float(spy_data['Close'].iloc[0])) / float(spy_data['Close'].iloc[0])) * 100
                
                spy_icon, spy_sentiment = get_spy_status_icon_and_label(spy_change, market_timeframe)
                change_color = "green" if spy_change > 0 else "red"
                
                st.markdown(f"""
                <div style="font-size: 1.5rem; font-weight: bold; margin: 5px 0;">${spy_close:,.2f}</div>
                <div style="color: {change_color}; font-size: 1rem; margin: 5px 0;">{spy_change:+.2f}%</div>
                <div style="font-size: 1rem; margin: 5px 0;">
                    <span style="margin-right: 5px;">{spy_icon}</span>
                    {spy_sentiment}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No SPY data available")
        except Exception as e:
            st.error(f"Error loading SPY data: {str(e)}")

    with col2:
        st.markdown("""
        <div style="font-size: 1.1rem; font-weight: bold; margin-bottom: 5px;">
            VIX Volatility Index
            <span class="tooltip">ℹ️
                <span class="tooltiptext">
                    <strong>VIX (Volatility Index)</strong><br>
                    🟢 Below 15 – Low volatility: Market confidence<br>
                    🟡 15–25 – Moderate volatility: Typical market movement<br>
                    🔴 Above 25 – High volatility: Fear & uncertainty
                </span>
            </span>
        </div>
        """, unsafe_allow_html=True)
        try:
            vix_data = yf.download("^VIX", period=yfinance_period, interval=yfinance_interval, progress=False)
            if not vix_data.empty and 'Close' in vix_data.columns:
                vix_close = float(vix_data['Close'].iloc[-1])
                vix_change = ((vix_close - float(vix_data['Close'].iloc[0])) / float(vix_data['Close'].iloc[0])) * 100
                
                if vix_close < 15:
                    vix_indicator = "🟢"
                    vix_status = "Low volatility"
                    change_color = "green" if vix_change < 0 else "red"
                elif 15 <= vix_close <= 25:
                    vix_indicator = "🟡"
                    vix_status = "Moderate volatility"
                    change_color = "#daa520"
                else:
                    vix_indicator = "🔴"
                    vix_status = "High volatility"
                    change_color = "red" if vix_change > 0 else "green"
                
                st.markdown(f"""
                <div style="font-size: 1.5rem; font-weight: bold; margin: 5px 0;">{vix_close:,.2f}</div>
                <div style="color: {change_color}; font-size: 1rem; margin: 5px 0;">{vix_change:+.2f}%</div>
                <div style="font-size: 1rem; margin: 5px 0;">
                    <span style="margin-right: 5px;">{vix_indicator}</span>
                    {vix_status}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No VIX data available")
        except Exception as e:
            st.error(f"Error loading VIX data: {str(e)}")

    with col3:
        st.markdown("""
        <div style="font-size: 1.1rem; font-weight: bold; margin-bottom: 5px;">
            10-Year Treasury Yield
            <span class="tooltip">ℹ️
                <span class="tooltiptext">
                    <strong>10-Year Treasury Yield</strong><br>
                    🟢 Below 3% – Accommodative: Easier borrowing, growth-friendly<br>
                    🟡 3–4% – Neutral: Balanced policy outlook<br>
                    🔴 Above 4% – Restrictive: Higher borrowing costs, market pressure
                </span>
            </span>
        </div>
        """, unsafe_allow_html=True)
        try:
            treasury_data = yf.download("^TNX", period=yfinance_period, interval=yfinance_interval, progress=False)
            if not treasury_data.empty and 'Close' in treasury_data.columns:
                yield_close = float(treasury_data['Close'].iloc[-1])
                yield_change = ((yield_close - float(treasury_data['Close'].iloc[0])) / float(treasury_data['Close'].iloc[0])) * 100
                
                if yield_close < 3:
                    yield_indicator = "🟢"
                    yield_status = "Accommodative"
                    change_color = "green" if yield_change < 0 else "red"
                elif 3 <= yield_close <= 4:
                    yield_indicator = "🟡"
                    yield_status = "Neutral"
                    change_color = "#daa520"
                else:
                    yield_indicator = "🔴"
                    yield_status = "Restrictive"
                    change_color = "red" if yield_change > 0 else "green"
                
                st.markdown(f"""
                <div style="font-size: 1.5rem; font-weight: bold; margin: 5px 0;">{yield_close:,.2f}%</div>
                <div style="color: {change_color}; font-size: 1rem; margin: 5px 0;">{yield_change:+.2f}%</div>
                <div style="font-size: 1rem; margin: 5px 0;">
                    <span style="margin-right: 5px;">{yield_indicator}</span>
                    {yield_status}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No yield data available")
        except Exception as e:
            st.error(f"Error loading yield data: {str(e)}")

    # --- Improved Sector Performance Section ---
    st.markdown("### 📊 Sector Performance")
    
    with st.spinner(f"Loading sector performance for {market_timeframe}..."):
        try:
            sector_perf = get_sector_performance_custom(market_timeframe)
            
            if sector_perf and len(sector_perf) > 0:
                # Create two columns with adjusted ratio (3:1)
                chart_col, table_col = st.columns([3, 1])
                
                with chart_col:
                    # Filter out None values and invalid data
                    valid_sectors = [(sector, perf) for sector, perf in sector_perf 
                                    if perf is not None and isinstance(perf, Number)]
                    
                    if valid_sectors:
                        # Sort by performance
                        valid_sectors.sort(key=lambda x: x[1], reverse=True)
                        
                        fig = px.bar(
                            x=[s[0] for s in valid_sectors],
                            y=[s[1] for s in valid_sectors],
                            color=[s[1] for s in valid_sectors],
                            color_continuous_scale="RdYlGn",
                            labels={"x": "", "y": "Return %"},
                            title=f"Sector Performance ({market_timeframe})",
                            height=450,
                            width=700,
                            text=[f"{s[1]:.1f}%" for s in valid_sectors]
                        )
                        fig.update_traces(
                            textposition='outside',
                            marker_line_color='rgba(0,0,0,0.5)',
                            marker_line_width=1,
                            textfont_size=12
                        )
                        fig.update_layout(
                            yaxis_tickformat=".1f",
                            margin=dict(l=20, r=20, t=40, b=20),
                            showlegend=False,  # Removed legend
                            xaxis_title=None,
                            yaxis_title="Return %",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(size=12),
                            hoverlabel=dict(
                                bgcolor="white",
                                font_size=12,
                                font_family="Arial"
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No valid sector performance data available")
                
                with table_col:
                    st.markdown("#### Performance Table")
                    
                    # Create dataframe with proper formatting
                    sector_df = pd.DataFrame(
                        [(s[0], f"{s[1]:.2f}%" if isinstance(s[1], Number) else "N/A") 
                         for s in sector_perf],
                        columns=["Sector", "Return %"]
                    )
                    
                    # Style the dataframe based on performance
                    def color_cells(val):
                        try:
                            num = float(val.replace('%', ''))
                            color = 'green' if num > 0 else 'red'
                            return f'color: {color}'
                        except:
                            return ''
                    
                    st.dataframe(
                        sector_df.style.applymap(color_cells, subset=['Return %']),
                        use_container_width=True,
                        height=450,  # Match chart height
                        hide_index=True
                    )
            else:
                st.warning("No sector performance data available")
                
        except Exception as e:
            st.error(f"Failed to load sector performance: {str(e)}")
            st.error(traceback.format_exc())

    # Add debug mode toggle in the Market Overview section
    if st.session_state.get('show_market_overview'):
        # Add debug mode toggle at the top
        debug_col, _ = st.columns([1, 5])
        with debug_col:
            st.session_state['debug_mode'] = st.checkbox("Debug Mode", value=False)

if not st.session_state.get('portfolio_created'):
    st.info("Enter tickers and click Apply to view your portfolio dashboard.")