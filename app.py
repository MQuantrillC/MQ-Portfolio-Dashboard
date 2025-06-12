import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.express as px
from collections import Counter
import traceback
import concurrent.futures
from functools import lru_cache
from numbers import Number
from typing import Tuple, Dict, List, Optional, Union
import random
import matplotlib
import plotly.colors
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import optional dependencies with fallbacks
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("scipy not available - some optimization features will be disabled")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.error("yfinance not available - please install with: pip install yfinance")

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    st.warning("requests/BeautifulSoup not available - web scraping features disabled")

try:
    from fp.fp import FreeProxy
    PROXY_AVAILABLE = True
except ImportError:
    PROXY_AVAILABLE = False
    # Don't show warning for proxy as it's optional

# Set page config
st.set_page_config(
    page_title=" Optimal Portfolio Dashboard",
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
if 'debug_mode' not in st.session_state:
    st.session_state['debug_mode'] = False
if 'selected_timeframe' not in st.session_state:
    st.session_state['selected_timeframe'] = '1M'
if 'risk_level' not in st.session_state:
    st.session_state['risk_level'] = 'Moderate'

# Custom CSS
st.markdown("""
<style>
    /* Main container adjustments */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Metric explanation styling */
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
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .company-info-metrics .element-container {font-size: 0.92em !important;}
</style>
""", unsafe_allow_html=True)

# Add custom CSS for enhanced styling
st.markdown('''
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.portfolio-header {
    background: linear-gradient(90deg, #4CAF50, #2196F3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}
</style>
''', unsafe_allow_html=True)

# Add custom CSS for tooltips
st.markdown('''
<style>
    /* Tooltip styling for dataframe cells */
    .stDataFrame [title] {
        position: relative;
    }
    .stDataFrame [title]:hover::after {
        content: attr(title);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background-color: #333;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        white-space: nowrap;
        z-index: 1000;
    }
</style>
''', unsafe_allow_html=True)

# Helper Functions
def format_value(value: Union[float, int], prefix: str = "$", suffix: str = "", decimals: int = 2) -> str:
    """Format numeric value with prefix, suffix, and decimal places"""
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        return f"{prefix}{value:,.{decimals}f}{suffix}"
    except Exception:
        return "N/A"

def remove_duplicates(items: List[str]) -> List[str]:
    """Remove duplicate items while preserving order"""
    return list(dict.fromkeys(items))

def get_proxy_dict(probability: float = 0.5) -> Optional[Dict[str, str]]:
    """Get a random proxy with given probability"""
    if not PROXY_AVAILABLE:
        return None
        
    if random.random() < probability:
        try:
            proxy = FreeProxy(rand=True).get()
            return {"http": proxy}
        except Exception as e:
            if st.session_state.get('debug_mode', False):
                st.error(f"Proxy error: {str(e)}")
            return None
    return None

def get_period_from_timeframe(timeframe: str) -> tuple:
    """Convert timeframe to yfinance period and interval"""
    timeframe_map = {
        '1D': ('1d', '1m'),
        '5D': ('5d', '5m'),
        '1M': ('1mo', '1d'),
        '6M': ('6mo', '1d'),  # Changed from '6mo' to ensure consistency
        '1Y': ('1y', '1d'),
        'YTD': ('ytd', '1d'),  # Explicitly use 'ytd' period
        '5Y': ('5y', '1wk'),
        '10Y': ('10y', '1mo')
    }
    return timeframe_map.get(timeframe, ('1mo', '1d'))

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_info(ticker: str) -> Optional[Dict]:
    """Fetch stock information with proxy rotation"""
    if not YFINANCE_AVAILABLE:
        st.error("yfinance is required for this feature")
        return None
        
    proxy = get_proxy_dict()
    try:
        if proxy:
            yf.set_config(proxy=proxy)
        else:
            yf.set_config(proxy=None)
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        if not info or "quoteType" not in info:
            return None
        return info
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Error fetching info for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_history(
    ticker: str,
    period: Optional[str] = None,
    interval: str = "1d",
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None
) -> Optional[pd.DataFrame]:
    """Fetch historical stock data with proxy rotation"""
    if not YFINANCE_AVAILABLE:
        st.error("yfinance is required for this feature")
        return None
        
    proxy = get_proxy_dict()
    try:
        if proxy:
            yf.set_config(proxy=proxy)
        else:
            yf.set_config(proxy=None)
        ticker_obj = yf.Ticker(ticker)
        
        # Convert datetime objects to strings if needed
        start_str = start.strftime('%Y-%m-%d') if isinstance(start, datetime) else start
        end_str = end.strftime('%Y-%m-%d') if isinstance(end, datetime) else end
        
        if start_str and end_str:
            hist = ticker_obj.history(start=start_str, end=end_str, interval=interval)
        elif period:
            hist = ticker_obj.history(period=period, interval=interval)
        else:
            return None
            
        if hist.empty:
            return None
            
        return hist
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Error fetching history for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_financial_statement(
    ticker: str,
    statement_type: str,
    period: str = "Annual"
) -> Optional[pd.DataFrame]:
    """Fetch financial statements with proxy rotation"""
    if not YFINANCE_AVAILABLE:
        st.error("yfinance is required for this feature")
        return None
        
    proxy = get_proxy_dict()
    try:
        if proxy:
            yf.set_config(proxy=proxy)
        else:
            yf.set_config(proxy=None)
        ticker_obj = yf.Ticker(ticker)
        
        if statement_type == "balance":
            data = ticker_obj.balance_sheet if period == "Annual" else ticker_obj.quarterly_balance_sheet
        elif statement_type == "income":
            data = ticker_obj.income_stmt if period == "Annual" else ticker_obj.quarterly_income_stmt
        elif statement_type == "cashflow":
            data = ticker_obj.cashflow if period == "Annual" else ticker_obj.quarterly_cashflow
        else:
            return None
            
        if data is None or data.empty:
            return None
            
        # Filter out columns with too many NaN values
        return data.loc[:, data.isna().mean() < 0.5]
        
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Error fetching {statement_type} for {ticker}: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_market_data(url: str) -> Optional[pd.DataFrame]:
    """Fetch market data from web with proxy rotation"""
    if not WEB_SCRAPING_AVAILABLE:
        st.error("requests and BeautifulSoup are required for this feature")
        return None
        
    proxy = get_proxy_dict()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, proxies=proxy, headers=headers, timeout=10)
        response.raise_for_status()
        tables = pd.read_html(response.content)
        if not tables:
            return None
        return tables[0]  # Return first table
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Error fetching market data: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_sector_performance() -> Optional[pd.DataFrame]:
    """Fetch sector performance data with proxy rotation"""
    if not YFINANCE_AVAILABLE:
        return None
        
    # Using SPDR sector ETFs as proxies
    sector_etfs = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLV": "Healthcare",
        "XLE": "Energy",
        "XLI": "Industrials",
        "XLP": "Consumer Staples",
        "XLY": "Consumer Discretionary",
        "XLB": "Materials",
        "XLU": "Utilities",
        "XLRE": "Real Estate"
    }
    
    performance_data = []
    for etf, sector in sector_etfs.items():
        hist = fetch_stock_history(etf, period="1mo")
        if hist is not None and not hist.empty:
            start_price = hist["Close"].iloc[0]
            end_price = hist["Close"].iloc[-1]
            change_pct = ((end_price - start_price) / start_price) * 100
            performance_data.append({
                "Sector": sector,
                "Change %": round(change_pct, 2),
                "ETF": etf
            })
    
    if not performance_data:
        return None
        
    df = pd.DataFrame(performance_data).sort_values("Change %", ascending=False)
    
    # Format the Change % column in the DataFrame with color coding
    def color_change(val):
        if val > 0:
            return f"color: green; font-weight: bold;"
        elif val < 0:
            return f"color: red; font-weight: bold;"
        return ""
    
    # Apply styling to the DataFrame
    styled_df = df.style.applymap(color_change, subset=['Change %'])
    styled_df = styled_df.format({'Change %': '{:.2f}%'})
    
    # Display the styled DataFrame
    st.dataframe(styled_df, use_container_width=True)
    
    # Create a bar chart with green-to-red gradient
    fig = px.bar(
        df,
        x='Sector',
        y='Change %',
        color='Change %',  # This will create the color scale
        color_continuous_scale=['red', 'lightgray', 'green'],  # Red to green
        color_continuous_midpoint=0,  # Neutral at 0%
        text='Change %',
        labels={'Change %': '1 Month Change (%)'},
        hover_data={'ETF': True}
    )
    
    # Customize the chart appearance
    fig.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside',
        marker_line_color='rgba(0,0,0,0.2)',
        marker_line_width=1
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title=None,
        yaxis_title='Change (%)',
        coloraxis_showscale=False,  # Hide the color scale legend
        height=500,
        hovermode='x unified',
        xaxis={'tickangle': 45},
        margin=dict(t=50, b=100)
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Add a note about the data source
    st.caption("""
        💡 Data based on SPDR sector ETFs. Performance shown is the 1-month percentage change.
        ETFs used: XLK (Tech), XLF (Financials), XLV (Healthcare), XLE (Energy), XLI (Industrials),
        XLP (Staples), XLY (Discretionary), XLB (Materials), XLU (Utilities), XLRE (Real Estate)
    """)
    
    return df

@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_sp500_tickers() -> List[str]:
    """Fetch S&P 500 ticker symbols"""
    if not WEB_SCRAPING_AVAILABLE:
        # Fallback list of popular S&P 500 stocks
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'UNH', 'JNJ',
            'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'PFE', 'ABBV', 'BAC',
            'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'DIS', 'ABT', 'WMT', 'CRM', 'MRK',
            'NFLX', 'ADBE', 'ACN', 'NKE', 'LLY', 'DHR', 'TXN', 'NEE', 'VZ', 'BMY',
            'QCOM', 'PM', 'T', 'UPS', 'RTX', 'SCHW', 'HON', 'LOW', 'AMD', 'AMGN'
        ]
    
    try:
        # Try to fetch from Wikipedia
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        tables = pd.read_html(response.content)
        if tables and len(tables) > 0:
            df = tables[0]
            # The first column usually contains the ticker symbols
            if 'Symbol' in df.columns:
                tickers = df['Symbol'].tolist()
            elif len(df.columns) > 0:
                tickers = df.iloc[:, 0].tolist()
            else:
                return []
            
            # Clean up tickers
            tickers = [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
            # Remove any invalid tickers
            tickers = [t for t in tickers if t and len(t) <= 5 and t != 'NAN']
            
            return sorted(tickers)
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Error fetching S&P 500 tickers: {str(e)}")
    
    # Return fallback list if fetching fails
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'UNH', 'JNJ',
        'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'PFE', 'ABBV', 'BAC',
        'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'DIS', 'ABT', 'WMT', 'CRM', 'MRK',
        'NFLX', 'ADBE', 'ACN', 'NKE', 'LLY', 'DHR', 'TXN', 'NEE', 'VZ', 'BMY',
        'QCOM', 'PM', 'T', 'UPS', 'RTX', 'SCHW', 'HON', 'LOW', 'AMD', 'AMGN'
    ]

def get_asset_price_change(ticker: str, period: str = "1mo") -> Optional[Dict[str, float]]:
    """Fetch asset's current price and % change over the given period."""
    df = fetch_stock_history(ticker, period=period)
    if df is not None and not df.empty:
        start = df['Close'].iloc[0]
        end = df['Close'].iloc[-1]
        pct_change = ((end - start) / start) * 100
        return {
            'current_price': end,
            'pct_change': pct_change
        }
    return None

def plot_gauge(value: float, min_val: float, max_val: float, title: str) -> go.Figure:
    """Create a gauge chart"""
    try:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [min_val, max_val/2], 'color': "lightgray"},
                    {'range': [max_val/2, max_val], 'color': "gray"}
                ]
            },
            title={'text': title}
        ))
        return fig
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Error creating gauge chart: {str(e)}")
        return None

def plot_candles_stick_bar(data: pd.DataFrame, title: str) -> go.Figure:
    """Create a candlestick chart"""
    try:
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])
        
        fig.update_layout(
            title=title,
            yaxis_title='Price',
            xaxis_title='Date',
            height=400
        )
        
        return fig
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Error creating candlestick chart: {str(e)}")
        return None

def info_table(info: Dict) -> pd.DataFrame:
    """Create a formatted info table from stock info"""
    try:
        if not info:
            return None
            
        # Select relevant fields
        fields = {
            'shortName': 'Name',
            'symbol': 'Symbol',
            'marketCap': 'Market Cap',
            'trailingPE': 'P/E Ratio',
            'forwardPE': 'Forward P/E',
            'dividendYield': 'Dividend Yield',
            'beta': 'Beta',
            'fiftyTwoWeekHigh': '52W High',
            'fiftyTwoWeekLow': '52W Low',
            'averageVolume': 'Avg Volume'
        }
        
        # Create table
        data = {display: info.get(field) for field, display in fields.items()}
        df = pd.DataFrame([data])
        
        # Format values
        if 'Market Cap' in df.columns:
            df['Market Cap'] = df['Market Cap'].apply(lambda x: format_value(x) if pd.notnull(x) else 'N/A')
        if 'Dividend Yield' in df.columns:
            df['Dividend Yield'] = df['Dividend Yield'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else 'N/A')
        if 'Avg Volume' in df.columns:
            df['Avg Volume'] = df['Avg Volume'].apply(lambda x: format_value(x, prefix='', suffix='', decimals=0) if pd.notnull(x) else 'N/A')
        
        return df
        
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Error creating info table: {str(e)}")
        return None

def top_table(data: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Create a top N table from DataFrame"""
    try:
        if data is None or data.empty:
            return None
        return data.head(n)
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Error creating top table: {str(e)}")
        return None

def analyze_portfolio(tickers: List[str]):
    """Analyze portfolio with basic info using selected year range"""
    with st.spinner("Analyzing portfolio..."):
        start_date = st.session_state.get('start_date')
        end_date = st.session_state.get('end_date')
        
        # Calculate per-ticker metrics
        returns = []
        volatilities = []
        change_pcts = []
        current_prices = []
        market_caps = []
        pe_ratios = []
        betas = []
        names = []
        
        for ticker in tickers:
            info = fetch_stock_info(ticker)
            hist = fetch_stock_history(
                ticker,
                start=start_date,
                end=end_date
            )
            
            if info and hist is not None and not hist.empty:
                # Calculate metrics
                daily_returns = hist["Close"].pct_change().dropna()
                annual_return = (1 + daily_returns.mean()) ** 252 - 1
                annual_volatility = daily_returns.std() * (252 ** 0.5)
                
                start_price = hist["Close"].iloc[0]
                end_price = hist["Close"].iloc[-1]
                pct_change = ((end_price - start_price) / start_price) * 100
                
                # Store metrics
                returns.append(annual_return)
                volatilities.append(annual_volatility)
                change_pcts.append(pct_change)
                current_prices.append(end_price)
                market_caps.append(info.get('marketCap'))
                pe_ratios.append(info.get('trailingPE'))
                betas.append(info.get('beta'))
                names.append(info.get('shortName', 'N/A'))
            else:
                # Handle missing data
                returns.append(None)
                volatilities.append(None)
                change_pcts.append(None)
                current_prices.append(None)
                market_caps.append(None)
                pe_ratios.append(None)
                betas.append(None)
                names.append('N/A')
        
        # Create portfolio data DataFrame
        portfolio_data = pd.DataFrame({
            'Ticker': tickers,
            'Name': names,
            'Current Price': [f"${price:.2f}" if price else 'N/A' for price in current_prices],
            'Expected Return': [f"{ret*100:.2f}%" if ret else 'N/A' for ret in returns],
            'Volatility': [f"{vol*100:.2f}%" if vol else 'N/A' for vol in volatilities],
            'Change %': [f"{chg:.2f}%" if chg else 'N/A' for chg in change_pcts],
            'Market Cap': [format_value(mc) if mc else 'N/A' for mc in market_caps],
            'P/E Ratio': [f"{pe:.2f}" if pe else 'N/A' for pe in pe_ratios],
            'Beta': [f"{beta:.2f}" if beta else 'N/A' for beta in betas]
        })
        
        if not portfolio_data.empty:
            st.dataframe(portfolio_data, use_container_width=True)
            return True, portfolio_data
        else:
            st.error("Could not fetch data for any stocks in the portfolio")
            return False, None

def get_optimized_weights(mean_returns, cov_matrix, risk_level):
    """Optimize portfolio weights based on risk level using modern portfolio theory."""
    n_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_volatility(weights, mean_returns, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
        port_return = np.dot(weights, mean_returns)
        port_vol = portfolio_volatility(weights, mean_returns, cov_matrix)
        return -(port_return - risk_free_rate) / port_vol if port_vol > 0 else 0

    def negative_return(weights, mean_returns):
        return -np.dot(weights, mean_returns)

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1. / n_assets] * n_assets)

    if risk_level == "Low":
        result = minimize(portfolio_volatility, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    elif risk_level == "Moderate":
        result = minimize(negative_sharpe, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    else:  # High
        result = minimize(negative_return, initial_weights, args=(mean_returns,), method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x if result.success else initial_weights

def color_performance(val):
    if isinstance(val, str) and '%' in val:
        try:
            num = float(val.replace('%', '').replace('+', ''))
            if num > 0:
                return 'background-color: rgba(0, 255, 0, 0.1); color: green'
            elif num < 0:
                return 'background-color: rgba(255, 0, 0, 0.1); color: red'
        except:
            pass
    return ''

def display_optimal_portfolio(tickers: List[str]):
    """Display optimal portfolio allocation and metrics based on selected risk level."""
    risk_level = st.session_state.get('risk_level', 'Moderate')
    start_date = st.session_state.get('start_date')
    end_date = st.session_state.get('end_date')
    
    # Risk level descriptions and icons
    risk_descriptions = {
        "Low": "🛡️ Low Risk - Minimizing Portfolio Volatility",
        "Moderate": "⚖️ Moderate Risk - Maximizing Sharpe Ratio", 
        "High": "🚀 High Risk - Maximizing Expected Returns",
        "Custom": "🎛️ Custom Risk - Manual Weight Selection"
    }
    
    # Show portfolio info in a single line
    st.info(f"**Portfolio:** {', '.join(tickers)} | **Strategy:** {risk_descriptions.get(risk_level, risk_level)} | **Year Range:** {start_date.year} to {end_date.year}")
    
    # Initialize weights and metrics
    weights = None
    metrics = None
    
    # Fetch historical data and calculate returns
    with st.spinner("Calculating portfolio metrics..."):
        returns_data = {}
        for ticker in tickers:
            hist = fetch_stock_history(
                ticker,
                start=start_date,
                end=end_date
            )
            if hist is not None and not hist.empty:
                returns_data[ticker] = hist['Close'].pct_change().dropna()
        
        if returns_data:
            returns_df = pd.DataFrame(returns_data)
            
            # Validate returns data
            if returns_df.empty:
                st.error("Not enough data to optimize. Please check tickers and time range.")
                st.stop()
            
            # Calculate annualized metrics
            mean_returns = returns_df.mean() * 252
            cov_matrix = returns_df.cov() * 252
            
            # Get weights based on risk level
            if risk_level == "Custom":
                # Get custom weights from session state
                custom_weights = st.session_state.get('custom_weights', {})
                total_weight = sum(custom_weights.values())
                
                # Validate custom weights
                if abs(total_weight - 100.0) > 0.01:
                    st.warning(f"Total custom weight = {total_weight:.2f}%. Must equal 100%.")
                    st.stop()
                
                # Convert custom weights to numpy array
                weights = np.array([custom_weights[ticker]/100.0 for ticker in tickers])
            else:
                # Optimize portfolio based on risk level
                with st.spinner(f"Optimizing portfolio for {risk_level} risk level..."):
                    weights = get_optimized_weights(mean_returns, cov_matrix, risk_level)
            
            # Calculate portfolio metrics
            if weights is not None:
                metrics = calculate_portfolio_metrics(returns_df, weights)
    
    # Display portfolio metrics at the top
    if metrics:
        st.markdown("### Portfolio Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Annual Return",
                f"{metrics['annual_return']*100:.2f}%",
                help="Expected annual return based on historical data"
            )
        with col2:
            st.metric(
                "Annual Volatility",
                f"{metrics['annual_vol']*100:.2f}%",
                help="Expected annual volatility (standard deviation of returns)"
            )
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.2f}",
                help="Risk-adjusted return (higher is better)"
            )
        with col4:
            st.metric(
                "Max Drawdown",
                f"{metrics['max_drawdown']*100:.2f}%",
                help="Largest historical percentage drop from peak"
            )
    
    # Display allocation details
    if weights is not None:
        # --- Parallel, cached info fetching ---
        with st.spinner("Fetching company info for all tickers..."):
            # Use session cache for all tickers
            cache = st.session_state.setdefault('stock_info_cache', {})
            missing = [t for t in tickers if t not in cache]
            if missing:
                fetched = fetch_all_stock_info(missing)
                cache.update(fetched)
            info_dict = {t: cache.get(t) for t in tickers}
        
        returns_data = {}
        for ticker in tickers:
            hist = fetch_stock_history(
                ticker,
                start=start_date,
                end=end_date
            )
            info = info_dict.get(ticker)
            if hist is not None and not hist.empty:
                daily_returns = hist["Close"].pct_change().dropna()
                annual_return = (1 + daily_returns.mean()) ** 252 - 1
                annual_volatility = daily_returns.std() * (252 ** 0.5)
                start_price = hist["Close"].iloc[0]
                end_price = hist["Close"].iloc[-1]
                pct_change = ((end_price - start_price) / start_price) * 100
                returns_data[ticker] = {
                    'Name': info.get('shortName', 'N/A') if info else 'N/A',
                    'Expected Return': f"{annual_return*100:.2f}%",
                    'Volatility': f"{annual_volatility*100:.2f}%",
                    'Change %': f"{pct_change:.2f}%",
                    'Market Cap': format_value(info.get('marketCap')) if info else 'N/A',
                    'P/E Ratio': f"{info.get('trailingPE', 'N/A')}" if info else 'N/A',
                    'Beta': f"{info.get('beta', 'N/A')}" if info else 'N/A'
                }
            else:
                returns_data[ticker] = {
                    'Name': info.get('shortName', 'N/A') if info else 'N/A',
                    'Expected Return': 'N/A',
                    'Volatility': 'N/A',
                    'Change %': 'N/A',
                    'Market Cap': format_value(info.get('marketCap')) if info else 'N/A',
                    'P/E Ratio': f"{info.get('trailingPE', 'N/A')}" if info else 'N/A',
                    'Beta': f"{info.get('beta', 'N/A')}" if info else 'N/A'
                }
        # Create portfolio data DataFrame
        portfolio_data = pd.DataFrame.from_dict(returns_data, orient='index')
        portfolio_data.index.name = 'Ticker'
        portfolio_data.reset_index(inplace=True)
        # Add weights to the portfolio data
        portfolio_data['Weight'] = [f"{w*100:.1f}%" for w in weights]
        # Add Current Price and Start of Period Price columns
        current_prices = []
        start_prices = []
        for ticker in portfolio_data['Ticker']:
            hist = fetch_stock_history(ticker, start=start_date, end=end_date)
            if hist is not None and not hist.empty:
                start_prices.append(hist['Close'].iloc[0])
                current_prices.append(hist['Close'].iloc[-1])
            else:
                start_prices.append(float('nan'))
                current_prices.append(float('nan'))
        portfolio_data.insert(2, 'Current Price', [f"${p:.2f}" if not pd.isna(p) else 'N/A' for p in current_prices])
        portfolio_data.insert(3, 'Start of Period Price', [f"${p:.2f}" if not pd.isna(p) else 'N/A' for p in start_prices])
        # Reorder columns for better display
        display_cols = ['Ticker', 'Name', 'Current Price', 'Start of Period Price', 'Weight', 'Expected Return', 'Volatility', 'Change %', 'Market Cap', 'P/E Ratio', 'Beta']
        portfolio_data = portfolio_data[display_cols]
        st.dataframe(portfolio_data, use_container_width=True)
        # Add subtle help text below the table (unchanged)
        st.markdown('''
        <div style="font-size: 0.85em; color: #666; margin-top: -15px; margin-bottom: 20px;">
            <b>Expected Return:</b> Annualized return since start date (higher is better) • 
            <b>Volatility:</b> Annualized standard deviation (lower is better) • 
            <b>Change %:</b> Total percentage change since start date (higher is better)
        </div>
        ''', unsafe_allow_html=True)

def create_portfolio_chart(tickers: List[str], timeframe: str):
    """Create portfolio performance chart using the selected timeframe."""
    # Convert timeframe to date range
    end_date = datetime.today()
    if timeframe == '1D':
        start_date = end_date - timedelta(days=1)
        interval = '1m'
    elif timeframe == '5D':
        start_date = end_date - timedelta(days=5)
        interval = '5m'
    elif timeframe == '1M':
        start_date = end_date - timedelta(days=30)
        interval = '1d'
    elif timeframe == '6M':
        start_date = end_date - timedelta(days=180)
        interval = '1d'
    elif timeframe == '1Y':
        start_date = end_date - timedelta(days=365)
        interval = '1d'
    elif timeframe == 'YTD':
        start_date = datetime(end_date.year, 1, 1)
        interval = '1d'
    elif timeframe == '5Y':
        start_date = end_date - timedelta(days=5*365)
        interval = '1wk'
    else:  # 10Y
        start_date = end_date - timedelta(days=10*365)
        interval = '1mo'
    with st.spinner(f"Loading chart data for {timeframe}..."):
        fig = go.Figure()
        # Use a qualitative palette for clear differentiation
        palette = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
        for i, ticker in enumerate(tickers):
            hist = fetch_stock_history(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval
            )
            if hist is not None and not hist.empty:
                normalized = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=normalized,
                    mode='lines',
                    name=ticker,
                    line=dict(width=2, color=palette[i % len(palette)]),
                    hovertemplate=f'<b>{ticker}</b><br>Date: %{{x}}<br>Change: %{{y:.2f}}%<extra></extra>'
                ))
        fig.update_layout(
            title=f"Portfolio Performance - {timeframe} (%)",
            xaxis_title="Date",
            yaxis_title="Change (%)",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # Additional features section - only show this at the bottom
    st.subheader("Additional Features")
    # Create tabs for additional features
    tab1, tab2, tab3, tab4 = st.tabs(["Key Ticker Information", "Financials", "Monte Carlo Simulation", "Market Overview"])

    with tab1:
        if st.session_state.get('portfolio_created', False):
            tickers = st.session_state.get('portfolio_tickers', [])
            st.subheader("Key Ticker Information")
            tabs = st.tabs(tickers)
            for tab, ticker in zip(tabs, tickers):
                with tab:
                    display_ticker_info(ticker)

    with tab2:
        if st.session_state.get('portfolio_created', False):
            tickers = st.session_state.get('portfolio_tickers', [])
            st.subheader("Financial Statements")
            
            # Define metric mappings for each statement type
            income_metric_map = [
                ("Total Revenue", ["total revenue", "totalrevenue"]),
                ("Cost of Revenue", ["cost of revenue", "costofrevenue"]),
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
            
            # Create tabs for each ticker
            ticker_tabs = st.tabs(tickers)
            
            for ticker_tab, ticker in zip(ticker_tabs, tickers):
                with ticker_tab:
                    st.markdown(f"### {ticker} Financial Statements")
                    
                    # Create inner tabs for each statement type
                    statement_tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow", "Key Ratios"])
                    
                    # Income Statement Tab
                    with statement_tabs[0]:
                        with st.spinner(f"Loading {ticker} Income Statement..."):
                            income_stmt = fetch_financial_statement(ticker, "income", "Annual")
                            if income_stmt is not None and not income_stmt.empty:
                                # Initialize filtered_data with correct columns
                                filtered_data = pd.DataFrame(columns=income_stmt.columns)
                                for display_name, possible_keys in income_metric_map:
                                    for key in possible_keys:
                                        if key.lower() in [k.lower() for k in income_stmt.index]:
                                            matching_key = next(k for k in income_stmt.index if k.lower() == key.lower())
                                            filtered_data.loc[display_name] = income_stmt.loc[matching_key]
                                            break
                                
                                if not filtered_data.empty:
                                    display_financial_analysis(filtered_data, "income", "Total Revenue")
                                else:
                                    st.warning(f"No matching metrics found for {ticker} Income Statement")
                            else:
                                st.error(f"Could not fetch Income Statement for {ticker}")
                    
                    # Balance Sheet Tab
                    with statement_tabs[1]:
                        with st.spinner(f"Loading {ticker} Balance Sheet..."):
                            balance_sheet = fetch_financial_statement(ticker, "balance", "Annual")
                            if balance_sheet is not None and not balance_sheet.empty:
                                # Initialize filtered_data with correct columns
                                filtered_data = pd.DataFrame(columns=balance_sheet.columns)
                                for display_name, possible_keys in balance_metric_map:
                                    for key in possible_keys:
                                        if key.lower() in [k.lower() for k in balance_sheet.index]:
                                            matching_key = next(k for k in balance_sheet.index if k.lower() == key.lower())
                                            filtered_data.loc[display_name] = balance_sheet.loc[matching_key]
                                            break
                                
                                if not filtered_data.empty:
                                    display_financial_analysis(filtered_data, "balance", "Total Assets")
                                else:
                                    st.warning(f"No matching metrics found for {ticker} Balance Sheet")
                            else:
                                st.error(f"Could not fetch Balance Sheet for {ticker}")
                    
                    # Cash Flow Tab
                    with statement_tabs[2]:
                        with st.spinner(f"Loading {ticker} Cash Flow Statement..."):
                            cash_flow = fetch_financial_statement(ticker, "cashflow", "Annual")
                            if cash_flow is not None and not cash_flow.empty:
                                # Initialize filtered_data with correct columns
                                filtered_data = pd.DataFrame(columns=cash_flow.columns)
                                for display_name, possible_keys in cashflow_metric_map:
                                    for key in possible_keys:
                                        if key.lower() in [k.lower() for k in cash_flow.index]:
                                            matching_key = next(k for k in cash_flow.index if k.lower() == key.lower())
                                            filtered_data.loc[display_name] = cash_flow.loc[matching_key]
                                            break
                                
                                if not filtered_data.empty:
                                    display_financial_analysis(filtered_data, "cashflow", "Operating Cash Flow")
                                else:
                                    st.warning(f"No matching metrics found for {ticker} Cash Flow Statement")
                            else:
                                st.error(f"Could not fetch Cash Flow Statement for {ticker}")
                    
                    # Key Ratios Tab
                    with statement_tabs[3]:
                        with st.spinner(f"Calculating {ticker} Key Ratios..."):
                            if all(df is not None and not df.empty for df in [income_stmt, balance_sheet, cash_flow]):
                                display_financial_ratios(ticker, income_stmt, balance_sheet, cash_flow)
                            else:
                                st.warning(f"Could not calculate key ratios for {ticker}. Some financial statements are missing.")
    with tab3:
        st.markdown("### Monte Carlo Simulation")
        st.markdown("""
        This Monte Carlo simulation projects potential future portfolio values based on:
        - Your portfolio's historical return and volatility
        - Random market fluctuations modeled with geometric Brownian motion

        The simulation runs multiple possible scenarios to show the range of potential outcomes.
        """)
        
        # Get portfolio metrics from session state
        start_date = st.session_state.get('start_date')
        end_date = st.session_state.get('end_date')
        risk_level = st.session_state.get('risk_level', 'Moderate')
        
        # Fetch historical data and calculate returns
        returns_data = {}
        for ticker in tickers:
            hist = fetch_stock_history(ticker, start=start_date, end=end_date)
            if hist is not None and not hist.empty:
                returns_data[ticker] = hist['Close'].pct_change().dropna()
        
        if not returns_data:
            st.error("Could not fetch historical data for simulation")
            return
            
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate annualized metrics
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        # Get weights based on risk level
        if risk_level == "Custom":
            custom_weights = st.session_state.get('custom_weights', {})
            weights = np.array([custom_weights[ticker]/100.0 for ticker in tickers])
        else:
            weights = get_optimized_weights(mean_returns, cov_matrix, risk_level)
        
        # Calculate portfolio metrics
        metrics = calculate_portfolio_metrics(returns_df, weights)
        
        if not metrics:
            st.error("Could not calculate portfolio metrics for simulation")
            return
            
        # Simulation parameters
        col1, col2 = st.columns(2)
        with col1:
            start_value = st.number_input("Starting Portfolio Value ($)", 
                                        min_value=1000, 
                                        max_value=10000000, 
                                        value=10000, 
                                        step=1000)
        with col2:
            years = st.slider("Time Horizon (Years)", 
                             min_value=1, 
                             max_value=30, 
                             value=10)
        
        simulations = st.slider("Number of Simulations", 
                               min_value=100, 
                               max_value=5000, 
                               value=1000, 
                               step=100)
        
        if st.button("Run Simulation", use_container_width=True):
            with st.spinner("Running Monte Carlo simulation..."):
                results = monte_carlo_simulation(
                    start_value=start_value,
                    mean_return=metrics['annual_return'],
                    volatility=metrics['annual_vol'],
                    years=years,
                    simulations=simulations
                )
                
                # Plot simulation paths
                fig = go.Figure()
                for i in range(min(100, simulations)):  # Plot max 100 paths for clarity
                    fig.add_trace(go.Scatter(
                        y=results[i],
                        mode='lines',
                        line=dict(width=1, color='rgba(100, 100, 255, 0.2)'),
                        showlegend=False,
                        hoverinfo='none'
                    ))
                
                # Add percentiles
                percentiles = np.percentile(results, [5, 50, 95], axis=0)
                fig.add_trace(go.Scatter(
                    y=percentiles[1],  # Median
                    mode='lines',
                    line=dict(width=3, color='blue'),
                    name='Median',
                    hovertemplate='Median: %{y:,.0f}<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    y=percentiles[0],  # 5th percentile
                    mode='lines',
                    line=dict(width=2, color='red', dash='dash'),
                    name='5th Percentile',
                    hovertemplate='Worst 5%%: %{y:,.0f}<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    y=percentiles[2],  # 95th percentile
                    mode='lines',
                    line=dict(width=2, color='green', dash='dash'),
                    name='95th Percentile',
                    hovertemplate='Best 5%%: %{y:,.0f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"Monte Carlo Simulation ({years} Years, {simulations} Runs)",
                    xaxis_title="Time Steps",
                    yaxis_title="Portfolio Value ($)",
                    height=500,
                    hovermode="x unified",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display key statistics
                final_values = results[:, -1]
                
                def format_stats(value):
                    change_pct = (value - start_value) / start_value * 100
                    return f"${value:,.2f} ({change_pct:+.2f}%)"
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Median Final Value", 
                            format_stats(np.median(final_values)),
                            help="50th percentile outcome")
                with col2:
                    st.metric("5th Percentile (Worst Case)", 
                            format_stats(np.percentile(final_values, 5)),
                            help="Only 5% of outcomes will be worse than this")
                with col3:
                    st.metric("95th Percentile (Best Case)", 
                            format_stats(np.percentile(final_values, 95)),
                            help="Only 5% of outcomes will be better than this")
                
                # Histogram of final values
                st.markdown("##### Final Value Distribution")
                fig_hist = px.histogram(
                    x=final_values,
                    nbins=50,
                    labels={'x': 'Final Portfolio Value ($)'},
                    color_discrete_sequence=['blue']
                )
                fig_hist.update_layout(
                    height=400,
                    showlegend=False,
                    xaxis_title="Final Portfolio Value",
                    yaxis_title="Number of Simulations"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
    with tab4:
        display_market_overview()

def get_market_indicators() -> Dict[str, Dict[str, Union[float, str]]]:
    """Fetch key market indicators including S&P 500, VIX, and Treasury yields"""
    try:
        indicators = {}
        
        # Fetch SPY data for S&P 500
        spy_data = fetch_stock_history("SPY", period="1d")
        if spy_data is not None and not spy_data.empty:
            spy_close = spy_data['Close'].iloc[-1]
            spy_change = ((spy_close - spy_data['Open'].iloc[0]) / spy_data['Open'].iloc[0]) * 100
            indicators['S&P 500'] = {
                'value': spy_close,
                'change': spy_change,
                'change_pct': f"{spy_change:+.2f}%",
                'icon': "🟢" if spy_change >= 0 else "🔴"
            }
        
        # Fetch VIX data
        vix_data = fetch_stock_history("^VIX", period="1d")
        if vix_data is not None and not vix_data.empty:
            vix_close = vix_data['Close'].iloc[-1]
            vix_change = ((vix_close - vix_data['Open'].iloc[0]) / vix_data['Open'].iloc[0]) * 100
            indicators['VIX'] = {
                'value': vix_close,
                'change': vix_change,
                'change_pct': f"{vix_change:+.2f}%",
                'icon': "🔴" if vix_change >= 0 else "🟢"  # Inverse for VIX
            }
        
        # Fetch 10Y Treasury Yield
        treasury_data = fetch_stock_history("^TNX", period="1d")
        if treasury_data is not None and not treasury_data.empty:
            treasury_close = treasury_data['Close'].iloc[-1]
            treasury_change = ((treasury_close - treasury_data['Open'].iloc[0]) / treasury_data['Open'].iloc[0]) * 100
            indicators['10Y Yield'] = {
                'value': treasury_close,
                'change': treasury_change,
                'change_pct': f"{treasury_change:+.2f}%",
                'icon': "🟢" if treasury_change >= 0 else "🔴"
            }
        
        return indicators
        
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Error fetching market indicators: {str(e)}")
        return {}

def create_metric_card(title, value, change, color):
    return f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}11);
        border-left: 4px solid {color};
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <h4 style="margin: 0; color: {color};">{title}</h4>
        <h2 style="margin: 0.5rem 0;">{value}</h2>
        <p style="margin: 0; font-size: 0.9rem; color: {color};">{change}</p>
    </div>
    """

def get_spy_status_icon_and_label(change_pct: float, timeframe: str) -> Tuple[str, str]:
    """Determine SPY status icon and label based on performance and timeframe."""
    if timeframe == "5D":
        if change_pct >= 1.5:
            return "🟢", "Bullish Week"
        elif change_pct <= -1.5:
            return "🔴", "Bearish Week"
        else:
            return "🟡", "Sideways"
    elif timeframe == "1M":
        if change_pct >= 3:
            return "🟢", "Strong Rally"
        elif change_pct <= -3:
            return "🔴", "Down Month"
        else:
            return "🟡", "Neutral"
    elif timeframe == "6M":  # Added 6M timeframe
        if change_pct >= 10:
            return "🟢", "Strong Rally"
        elif change_pct <= -10:
            return "🔴", "Downtrend"
        else:
            return "🟡", "Moderate"
    elif timeframe == "YTD":  # Added YTD timeframe
        if change_pct >= 15:
            return "🟢", "Strong YTD"
        elif change_pct <= -15:
            return "🔴", "Weak YTD"
        else:
            return "🟡", "Neutral YTD"
    elif timeframe == "1Y":
        if change_pct >= 8:
            return "🟢", "Strong Year"
        elif change_pct <= -8:
            return "🔴", "Bearish Year"
        else:
            return "🟡", "Flat Year"
    elif timeframe == "5Y":
        if change_pct >= 35:
            return "🟢", "Strong Growth"
        elif change_pct <= 5:
            return "🔴", "Underperforming"
        else:
            return "🟡", "Moderate"
    elif timeframe == "10Y":
        if change_pct >= 80:
            return "🟢", "Strong Decade"
        elif change_pct <= 10:
            return "🔴", "Weak Decade"
        else:
            return "🟡", "Flat Decade"
    else:
        return "ℹ️", "Normal" if change_pct >= 0 else "⚠️", "Caution"

def fetch_data_for_timeframe(ticker: str, mo_period: Optional[str], mo_interval: str, market_tf: str) -> Optional[pd.DataFrame]:
    """Fetch stock data with proper handling of YTD and other timeframes."""
    if market_tf == "YTD":
        current_year = datetime.today().year
        start_date = datetime(current_year, 1, 1)
        return fetch_stock_history(ticker, start=start_date, end=datetime.today(), interval=mo_interval)
    else:
        return fetch_stock_history(ticker, period=mo_period, interval=mo_interval)

def display_market_overview():
    st.markdown("## Market Overview")

    # Timeframe selector - remove 1D option
    market_tf = st.selectbox(
        "Select Market Overview Timeframe",
        options=["5D", "1M", "6M", "1Y", "YTD", "5Y", "10Y"],  # Removed "1D"
        index=1,  # Default to 1M
        key="market_overview_timeframe"
    )
    
    # Get period and interval
    mo_period, mo_interval = get_period_from_timeframe(market_tf)
    
    # For YTD, we need to calculate the start date as Jan 1 of current year
    if market_tf == "YTD":
        current_year = datetime.today().year
        start_date = datetime(current_year, 1, 1)
        end_date = datetime.today()
        # Override the period/interval for YTD to use date range
        mo_period = None
        mo_interval = "1d"
    elif market_tf == "6M":
        # For 6 months, ensure we're getting daily data
        mo_period = "6mo"
        mo_interval = "1d"

    # Key indicators section
    st.markdown("### Key Market Indicators")
    col1, col2, col3 = st.columns(3)
    
    # --- SPY (S&P 500) ---
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
            "6M": """
                <strong>S&P 500 (SPY) 6-Month Performance</strong><br>
                🟢 Strong Rally: +10% or more<br>
                🔴 Downtrend: -10% or less<br>
                🟡 Moderate: Between thresholds
            """,
            "YTD": """
                <strong>S&P 500 (SPY) Year-to-Date Performance</strong><br>
                🟢 Strong YTD: +15% or more<br>
                🔴 Weak YTD: -15% or less<br>
                🟡 Neutral YTD: Between thresholds
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
                <span class="tooltiptext">{timeframe_tooltips.get(market_tf, "Performance metrics")}</span>
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        spy_data = fetch_data_for_timeframe("SPY", mo_period, mo_interval, market_tf)
        if spy_data is not None and not spy_data.empty:
            spy_close = spy_data['Close'].iloc[-1]
            spy_change = ((spy_close - spy_data['Close'].iloc[0]) / spy_data['Close'].iloc[0]) * 100
            
            spy_icon, spy_sentiment = get_spy_status_icon_and_label(spy_change, market_tf)
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

    # --- VIX ---
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
        
        vix_data = fetch_data_for_timeframe("^VIX", mo_period, mo_interval, market_tf)
        if vix_data is not None and not vix_data.empty:
            vix_close = vix_data['Close'].iloc[-1]
            vix_change = ((vix_close - vix_data['Close'].iloc[0]) / vix_data['Close'].iloc[0]) * 100
            
            if vix_close < 15:
                vix_indicator = "🟢"
                vix_status = "Low volatility"
                change_color = "green" if vix_change < 0 else "red"
            elif 15 <= vix_close <= 25:
                vix_indicator = "🟡"
                vix_status = "Moderate volatility"
                change_color = "#daa520"  # Gold color for moderate
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

    # --- 10-Year Treasury Yield ---
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
        
        treasury_data = fetch_data_for_timeframe("^TNX", mo_period, mo_interval, market_tf)
        if treasury_data is not None and not treasury_data.empty:
            yield_close = treasury_data['Close'].iloc[-1]
            yield_change = ((yield_close - treasury_data['Close'].iloc[0]) / treasury_data['Close'].iloc[0]) * 100
            
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

    # Rest of your existing market overview code (commodities, forex, crypto, sectors)
    st.markdown("### Commodities, Forex & Crypto")
    
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

    tab1, tab2, tab3 = st.tabs(["Commodities", "Forex", "Crypto"])

    def render_assets(assets, is_forex=False):
        cols = st.columns(len(assets))
        for i, (name, data) in enumerate(assets.items()):
            ticker = data if not is_forex else data[0]
            tooltip = None if not is_forex else data[1]
            with cols[i]:
                with st.spinner(f"Loading {name}..."):
                    result = get_asset_price_change(ticker, period=mo_period)
                    if result:
                        price = f"${result['current_price']:,.2f}" if "Index" not in name else f"{result['current_price']:.2f}"
                        change = result['pct_change']
                        color = "green" if change >= 0 else "red"
                        tooltip_html = ""
                        if is_forex:
                            formatted_rate = f"{result['current_price']:.4f}" if "Index" not in name else f"{result['current_price']:.2f}"
                            full_tooltip = tooltip.format(rate=formatted_rate)
                            if "Index" in name:
                                full_tooltip += "<br><br>🔴 Above 100 = USD strong<br>🟢 Below 100 = USD weak"
                            tooltip_html = f'''<span class="tooltip">ℹ️
                                <span class="tooltiptext">{full_tooltip}</span>
                            </span>'''
                            name = name.split("ℹ️")[0].strip()
                        st.markdown(f'''
                            <div style="font-weight:bold;">{name} {tooltip_html}</div>
                            <div style="font-size:1.4rem;">{price}</div>
                            <div style="color:{color};">{change:+.2f}%</div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ {name} data not available")

    with tab1:
        st.markdown("#### Commodities Performance")
        render_assets(commodities)

    with tab2:
        st.markdown("#### Forex Performance")
        render_assets(forex, is_forex=True)

    with tab3:
        st.markdown("#### Cryptocurrency Performance")
        render_assets(crypto)

    # --- Sector Performance ---
    st.markdown("### Sector Performance")
    with st.spinner("Loading sector performance..."):
        sector_df = fetch_sector_performance()
        if sector_df is not None and not sector_df.empty:
            st.dataframe(sector_df, use_container_width=True)
        else:
            st.warning("No sector data available.")

def calculate_portfolio_metrics(returns: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
    """Calculate key portfolio metrics"""
    try:
        # Portfolio returns
        portfolio_returns = returns.dot(weights)
        
        # Annualized metrics
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
        
        # Maximum drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        return {
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Error calculating portfolio metrics: {str(e)}")
        return {}

# Add this import and palette at the top (if not already present)
import plotly.express as px
custom_palette = px.colors.sequential.Blues + px.colors.sequential.Purples

def display_benchmark_comparison(tickers: List[str], weights: Optional[np.ndarray] = None):
    """Display portfolio performance comparison with S&P 500 using selected date range"""
    st.subheader("Portfolio vs. S&P 500 Benchmark")
    
    start_date = st.session_state.get('start_date')
    end_date = st.session_state.get('end_date')
    start_year = start_date.year if start_date else None
    end_year = end_date.year if end_date else None
    
    with st.spinner("Loading benchmark comparison..."):
        # Fetch portfolio data
        portfolio_data = {}
        for ticker in tickers:
            hist = fetch_stock_history(
                ticker,
                start=start_date,
                end=end_date
            )
            if hist is not None and not hist.empty:
                portfolio_data[ticker] = hist['Close']
        if not portfolio_data:
            st.error("Could not fetch portfolio data")
            return
        # Create portfolio value series
        portfolio_df = pd.DataFrame(portfolio_data)
        if weights is None:
            weights = np.array([1/len(tickers)] * len(tickers))  # Equal weights if not provided
        portfolio_value = portfolio_df.dot(weights)
        # Fetch SPY data
        if proxy:
            yf.set_config(proxy=proxy)
        else:
            yf.set_config(proxy=None)
        spy_data = yf.Ticker("SPY")
        if spy_data is None or spy_data.empty:
            st.error("Could not fetch S&P 500 data")
            return
        # Normalize both series to 100
        portfolio_normalized = portfolio_value / portfolio_value.iloc[0] * 100
        spy_normalized = spy_data.history(period="1d")['Close'] / spy_data.history(period="1d")['Close'].iloc[0] * 100
        # --- Tabs for main chart and yearly breakdown ---
        tab1, tab2 = st.tabs(["Performance Chart", "Yearly Performance Breakdown"])
        with tab1:
            # Create comparison chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_normalized.index,
                y=portfolio_normalized,
                mode='lines',
                name='Portfolio',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=spy_normalized.index,
                y=spy_normalized,
                mode='lines',
                name='S&P 500',
                line=dict(color='gray', dash='dash')
            ))
            # Calculate date range for title
            date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            fig.update_layout(
                title=f'Portfolio vs. S&P 500 Performance ({date_range})',
                xaxis_title='Date',
                yaxis_title='Value (Normalized to 100)',
                height=500,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            # Calculate annual returns for each ticker
            annual_returns = []
            for ticker in tickers:
                hist = fetch_stock_history(
                    ticker,
                    start=start_date,
                    end=end_date
                )
                if hist is not None and not hist.empty:
                    hist = hist.resample('Y').last()
                    for i in range(1, len(hist)):
                        year = hist.index[i].year
                        prev_price = hist['Close'].iloc[i-1]
                        curr_price = hist['Close'].iloc[i]
                        ret = (curr_price - prev_price) / prev_price * 100
                        if (not start_year or year >= start_year) and (not end_year or year <= end_year):
                            annual_returns.append({
                                'Year': year,
                                'Ticker': ticker,
                                'Annual Return (%)': ret
                            })
            if annual_returns:
                df_annual = pd.DataFrame(annual_returns)
                df_annual = df_annual.sort_values(['Year', 'Ticker'])
                # Add a color column for text based on value
                df_annual['TextColor'] = df_annual['Annual Return (%)'].apply(lambda x: 'green' if x >= 0 else 'red')
                fig = px.bar(
                    df_annual,
                    x='Year',
                    y='Annual Return (%)',
                    color='Ticker',
                    barmode='group',
                    color_discrete_sequence=px.colors.qualitative.Set3,  # Vibrant colors
                    title='Yearly Performance Breakdown',
                    hover_data={'Annual Return (%)': ':.2f%'},
                    text='Annual Return (%)'
                )
                fig.update_traces(
                    marker_line_color='rgba(0,0,0,0.2)',
                    marker_line_width=1,
                    opacity=0.8,
                    texttemplate='%{text:.2f}%',
                    textposition='outside',
                )
                # Set text color by value (green/red)
                for i, d in enumerate(fig.data):
                    # Get mask for this ticker
                    mask = df_annual['Ticker'] == d.name
                    colors = df_annual.loc[mask, 'TextColor'].tolist()
                    d.textfont = dict(color=colors)
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    title_font_size=16,
                    xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to display yearly performance breakdown.")

def display_ticker_info(ticker: str):
    """Display detailed information for a single ticker"""
    info = fetch_stock_info(ticker)
    if not info:
        st.error(f"Could not fetch information for {ticker}")
        return
    
    # Company Information
    st.markdown("### Company Information")
    # Reduce font size for company info using HTML/CSS
    st.markdown("""
    <style>
    .company-info-metrics .element-container {font-size: 0.92em !important;}
    </style>
    """, unsafe_allow_html=True)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Company Name", info.get('shortName', 'N/A'))
            st.metric("CEO", info.get('companyOfficers', [{}])[0].get('name', 'N/A') if info.get('companyOfficers') else 'N/A')
            st.metric("Employees", format_value(info.get('fullTimeEmployees', 0), prefix='', suffix='', decimals=0))
        with col2:
            st.metric("Sector", info.get('sector', 'N/A'))
            st.metric("Industry", info.get('industry', 'N/A'))
            st.metric("Headquarters", f"{info.get('city', 'N/A')}, {info.get('state', 'N/A')}")
        # Add current stock price as a metric (spanning both columns)
        st.metric("Current Stock Price", format_value(info.get('currentPrice', None)))
    
    # Historical Prices
    st.markdown("### Historical Prices")
    # Updated period labels
    periods = {
        '1 Month Ago': '1mo',
        '6 Months Ago': '6mo',
        '1 Year Ago': '1y',
        '5 Years Ago': '5y'
    }
    price_data = []
    for period_name, period in periods.items():
        hist = fetch_stock_history(ticker, period=period)
        if hist is not None and not hist.empty:
            current_price = hist['Close'].iloc[-1]
            start_price = hist['Close'].iloc[0]
            change_pct = ((current_price - start_price) / start_price) * 100
            price_data.append({
                'Period': period_name,
                'Price': format_value(start_price),
                'Change': f"{change_pct:+.2f}%"
            })
    if price_data:
        df = pd.DataFrame(price_data)
        st.dataframe(df.style.set_properties(**{'font-size': '13px'}), use_container_width=True)
    # 52 Week High/Low
    st.markdown("### 52 Week Range")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("52 Week High", format_value(info.get('fiftyTwoWeekHigh')))
    with col2:
        st.metric("52 Week Low", format_value(info.get('fiftyTwoWeekLow')))

def format_financial_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Simplify financial columns to just year labels (e.g., 2022) and sort descending.
    
    Args:
        df: DataFrame with datetime or string column names
        
    Returns:
        DataFrame with:
        - Only columns with valid years included
        - Columns sorted by year in descending order (newest to oldest)
        - Column names simplified to just years (e.g., '2024' instead of '2024-09-30')
    """
    if df is None or df.empty:
        return None

    # Extract valid year mappings
    year_cols = {}
    for col in df.columns:
        try:
            if isinstance(col, (pd.Timestamp, datetime)):
                # Check if this column has any non-null data
                if not df[col].isna().all():
                    year = col.year
                    year_cols[col] = year
            else:
                # For string columns, try to extract year and check for data
                if not df[col].isna().all():
                    year_str = str(col).split("-")[0]
                    if year_str.isdigit():
                        year = int(year_str)
                        year_cols[col] = year
        except (ValueError, AttributeError):
            continue

    # Keep only columns with valid year mapping AND actual data
    if not year_cols:
        return None

    # Sort by year descending and get the original column names
    sorted_cols = sorted(year_cols.items(), key=lambda x: x[1], reverse=True)
    sorted_original_cols = [col for col, year in sorted_cols]

    # Rename columns to just years
    df_sorted = df[sorted_original_cols].copy()
    df_sorted.columns = [str(year_cols[col]) for col in sorted_original_cols]
    
    return df_sorted

def calculate_horizontal_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate year-over-year percentage changes for each metric."""
    if df is None or df.empty or len(df.columns) < 2:
        return None
    
    # First, ensure columns are properly sorted by year (newest to oldest)
    df_sorted = format_financial_columns(df)
    if df_sorted is None or len(df_sorted.columns) < 2:
        return None
    
    # Explicitly sort years in reverse order (newest to oldest)
    years = sorted([int(col) for col in df_sorted.columns], reverse=True)
    df_sorted = df_sorted[[str(year) for year in years]]
    
    # Calculate percentage changes only between consecutive years with data
    pct_changes = pd.DataFrame(index=df_sorted.index)
    
    for i in range(len(df_sorted.columns)-1):
        current_col = df_sorted.columns[i]  # Newer year
        prev_col = df_sorted.columns[i+1]   # Older year
        
        # Only calculate if both years have non-null data
        valid_rows = ~df_sorted[current_col].isna() & ~df_sorted[prev_col].isna()
        if valid_rows.any():
            changes = (df_sorted.loc[valid_rows, current_col] - df_sorted.loc[valid_rows, prev_col]) / df_sorted.loc[valid_rows, prev_col].abs() * 100
            pct_changes.loc[valid_rows, current_col] = changes
    
    # Format with colors and arrows
    def format_change(val):
        if pd.isna(val):
            return 'N/A'
        color = 'green' if val > 0 else 'red'
        arrow = '▲' if val > 0 else '▼'
        return f'<span style="color: {color}">{arrow} {val:+.1f}%</span>'
    
    return pct_changes.applymap(format_change) if not pct_changes.empty else None

def calculate_vertical_analysis(df: pd.DataFrame, base_metric: str, statement_type: str) -> pd.DataFrame:
    """Calculate vertical analysis (percentages of base metric) for each year."""
    if df is None or df.empty:
        return None
        
    # Format and sort columns
    df_sorted = format_financial_columns(df)
    if df_sorted is None:
        return None
    
    # Get base metric values - make lookup more flexible
    base_values = None
    possible_names = [base_metric.lower()]
    
    # Add alternative names based on statement type
    if statement_type == "income":
        possible_names.extend([
            "revenue", "totalrevenue", "sales", "total sales", "total revenue",
            "net sales", "net revenue", "gross sales", "gross revenue"
        ])
    elif statement_type == "balance":
        possible_names.extend([
            "totalassets", "total assets", "assets", "total current assets",
            "total non-current assets", "total long-term assets"
        ])
    elif statement_type == "cashflow":
        possible_names.extend([
            "operating cash flow", "cash from operations", "net cash provided by operating activities",
            "total cash from operating activities", "cash flow from operations"
        ])
    
    # Try to find a matching metric
    for idx in df_sorted.index:
        idx_lower = str(idx).lower()
        if any(name in idx_lower for name in possible_names):
            base_values = df_sorted.loc[idx]
            break
    
    if base_values is None:
        return None
    
    # Calculate percentages
    percentages = df_sorted.div(base_values) * 100
    
    # Format percentages
    def format_percentage(val):
        if pd.isna(val):
            return 'N/A'
        return f'{val:.1f}%'
    
    formatted_percentages = percentages.applymap(format_percentage)
    return formatted_percentages

def style_financial_analysis(df: pd.DataFrame, analysis_type: str) -> pd.DataFrame:
    """Apply styling to financial analysis DataFrame."""
    if df is None or df.empty:
        return None
    
    # Define styles based on analysis type
    if analysis_type == 'horizontal':
        # Green for positive, red for negative
        def style_horizontal(val):
            if isinstance(val, str):
                if '▲' in val:
                    return 'background-color: rgba(0, 255, 0, 0.1); color: green'
                elif '▼' in val:
                    return 'background-color: rgba(255, 0, 0, 0.1); color: red'
            return ''
        return df.style.applymap(style_horizontal)
    else:  # vertical
        # Blue gradient based on value
        def style_vertical(val):
            if isinstance(val, str) and '%' in val:
                try:
                    num = float(val.replace('%', ''))
                    # Normalize to 0-1 range for color intensity
                    intensity = min(num / 100, 1)
                    return f'background-color: rgba(0, 0, 255, {intensity * 0.1})'
                except:
                    pass
            return ''
        return df.style.applymap(style_vertical)

def display_financial_analysis(df: pd.DataFrame, statement_type: str, base_metric: str):
    """Display horizontal and vertical analysis for a financial statement."""
    if df is None or df.empty:
        st.warning("No data available for analysis")
        return
    
    # Format the main data table
    df_formatted = format_financial_columns(df)
    if df_formatted is not None:
        st.dataframe(df_formatted.style.format(lambda x: format_value(x) if pd.notnull(x) else 'N/A'))
    
    # Horizontal Analysis
    st.markdown("### Horizontal Analysis")
    horizontal_df = calculate_horizontal_analysis(df)
    if horizontal_df is not None and not horizontal_df.empty:
        if len(df.columns) < 2:
            st.info("Horizontal analysis requires at least 2 years of data")
        else:
            styled_horizontal = style_financial_analysis(horizontal_df, 'horizontal')
            st.markdown(styled_horizontal.to_html(escape=False), unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size: 0.85em; color: #666; margin-top: -15px; margin-bottom: 20px;">
                ▲ Green: Positive year-over-year change • ▼ Red: Negative year-over-year change • 
                Columns show changes from previous year (e.g., 2024 shows change from 2023)
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Not enough data for horizontal analysis")
    
    # Vertical Analysis
    st.markdown("### Vertical Analysis")
    vertical_df = calculate_vertical_analysis(df, base_metric, statement_type)
    if vertical_df is not None and not vertical_df.empty:
        styled_vertical = style_financial_analysis(vertical_df, 'vertical')
        st.markdown(styled_vertical.to_html(escape=False), unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size: 0.85em; color: #666; margin-top: -15px; margin-bottom: 20px;">
            Values shown as percentage of {base_metric}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"Could not calculate vertical analysis. Base metric '{base_metric}' not found or insufficient data.")

# Add helper functions for ratio calculations
def current_ratio(current_assets, current_liabilities):
    """Calculate current ratio with safe division."""
    return current_assets / current_liabilities if current_liabilities else None

def quick_ratio(current_assets, inventory, current_liabilities):
    """Calculate quick ratio with safe division."""
    return (current_assets - inventory) / current_liabilities if current_liabilities else None

def working_capital(current_assets, current_liabilities):
    """Calculate working capital with safe subtraction."""
    return current_assets - current_liabilities if current_assets is not None and current_liabilities is not None else None

def gross_margin(gross_profit, revenue):
    """Calculate gross margin with safe division."""
    return gross_profit / revenue if revenue else None

def operating_margin(operating_income, revenue):
    """Calculate operating margin with safe division."""
    return operating_income / revenue if revenue else None

def net_margin(net_income, revenue):
    """Calculate net margin with safe division."""
    return net_income / revenue if revenue else None

def return_on_assets(net_income, total_assets):
    """Calculate ROA with safe division."""
    return net_income / total_assets if total_assets else None

def return_on_equity(net_income, equity):
    """Calculate ROE with safe division."""
    return net_income / equity if equity else None

def dso(avg_receivables, revenue):
    """Calculate Days Sales Outstanding with safe division."""
    return (avg_receivables / revenue) * 365 if avg_receivables is not None and revenue else None

def dpo(avg_payables, purchases):
    """Calculate Days Payable Outstanding with safe division."""
    return (avg_payables / purchases) * 365 if avg_payables is not None and purchases else None

def dio(avg_inventory, cogs):
    """Calculate Days Inventory Outstanding with safe division."""
    return (avg_inventory / cogs) * 365 if avg_inventory is not None and cogs else None

def cash_conversion_cycle(dso_val, dio_val, dpo_val):
    """Calculate Cash Conversion Cycle with safe addition/subtraction."""
    if dso_val is not None and dio_val is not None and dpo_val is not None:
        return dso_val + dio_val - dpo_val
    return None

def calculate_financial_ratios(income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Calculate key financial ratios from financial statements with improved accuracy."""
    if any(df is None or df.empty for df in [income_stmt, balance_sheet, cash_flow]):
        return None
        
    # Format and sort all statements by year
    income_stmt = format_financial_columns(income_stmt)
    balance_sheet = format_financial_columns(balance_sheet)
    cash_flow = format_financial_columns(cash_flow)
    
    if any(df is None for df in [income_stmt, balance_sheet, cash_flow]):
        return None
    
    # Get common years across all statements
    years = sorted(list(set(income_stmt.columns) & set(balance_sheet.columns) & set(cash_flow.columns)), reverse=True)
    if not years:
        return None
    
    # Initialize ratio DataFrames
    liquidity_ratios = pd.DataFrame(index=['Current Ratio', 'Quick Ratio', 'Working Capital'], columns=years)
    profitability_ratios = pd.DataFrame(index=['Gross Margin', 'Operating Margin', 'Net Margin', 'ROA', 'ROE'], columns=years)
    efficiency_ratios = pd.DataFrame(index=['DSO', 'DPO', 'DIO', 'Cash Conversion Cycle'], columns=years)
    leverage_ratios = pd.DataFrame(index=['Debt/Equity', 'Debt/Assets'], columns=years)
    
    # Helper function to safely get values with multiple fallback options
    def get_value(df: pd.DataFrame, primary_name: str, fallback_names: List[str], year: str) -> float:
        """Get value with multiple fallback options and proper error handling."""
        names_to_try = [primary_name] + fallback_names
        for name in names_to_try:
            try:
                # Exact match
                if name in df.index:
                    return df.loc[name, year]
                
                # Case-insensitive partial match
                matches = [idx for idx in df.index if str(name).lower() in str(idx).lower()]
                if matches:
                    return df.loc[matches[0], year]
            except:
                continue
        return None
    
    # Calculate ratios for each year
    for i, year in enumerate(years):
        try:
            # Get key values with comprehensive fallback options
            current_assets = get_value(balance_sheet, 'Total Current Assets', 
                                     ['Current Assets'], year)
            current_liabilities = get_value(balance_sheet, 'Total Current Liabilities',
                                          ['Current Liabilities'], year)
            inventory = get_value(balance_sheet, 'Inventory',
                                ['Inventories'], year)
            total_assets = get_value(balance_sheet, 'Total Assets', [], year)
            total_equity = get_value(balance_sheet, "Total Stockholder Equity",
                                   ['Total Equity', 'Stockholders Equity'], year)
            total_debt = get_value(balance_sheet, 'Total Debt',
                                 ['Long Term Debt', 'Debt'], year)
            
            revenue = get_value(income_stmt, 'Total Revenue',
                              ['Revenue', 'Sales', 'Net Sales'], year)
            gross_profit = get_value(income_stmt, 'Gross Profit',
                                   ['Gross Income'], year)
            operating_income = get_value(income_stmt, 'Operating Income',
                                       ['Operating Profit'], year)
            net_income = get_value(income_stmt, 'Net Income',
                                 ['Net Earnings', 'Profit'], year)
            cogs = get_value(income_stmt, 'Cost Of Revenue',
                           ['Cost of Goods Sold', 'COGS'], year)
            
            # Get efficiency metrics - try to get average if possible
            receivables = get_value(balance_sheet, 'Accounts Receivable',
                                  ['Receivables', 'Net Receivables'], year)
            payables = get_value(balance_sheet, 'Accounts Payable',
                               ['Payables'], year)
            
            # Calculate Liquidity Ratios (safely handle division by zero)
            liquidity_ratios.loc['Current Ratio', year] = (
                current_assets / current_liabilities if current_liabilities and current_liabilities != 0 else None
            )
            liquidity_ratios.loc['Quick Ratio', year] = (
                (current_assets - inventory) / current_liabilities 
                if current_liabilities and current_liabilities != 0 else None
            )
            liquidity_ratios.loc['Working Capital', year] = (
                current_assets - current_liabilities 
                if current_assets is not None and current_liabilities is not None else None
            )
            
            # Calculate Profitability Ratios
            profitability_ratios.loc['Gross Margin', year] = (
                gross_profit / revenue if revenue and revenue != 0 else None
            )
            profitability_ratios.loc['Operating Margin', year] = (
                operating_income / revenue if revenue and revenue != 0 else None
            )
            profitability_ratios.loc['Net Margin', year] = (
                net_income / revenue if revenue and revenue != 0 else None
            )
            profitability_ratios.loc['ROA', year] = (
                net_income / total_assets if total_assets and total_assets != 0 else None
            )
            profitability_ratios.loc['ROE', year] = (
                net_income / total_equity if total_equity and total_equity != 0 else None
            )
            
            # Calculate Efficiency Ratios (using more accurate formulas)
            # For DSO: Accounts Receivable / (Revenue/365)
            dso_val = (
                (receivables / (revenue / 365)) 
                if revenue and revenue != 0 and receivables is not None else None
            )
            
            # For DPO: Accounts Payable / (COGS/365)
            dpo_val = (
                (payables / (cogs / 365)) 
                if cogs and cogs != 0 and payables is not None else None
            )
            
            # For DIO: Inventory / (COGS/365)
            dio_val = (
                (inventory / (cogs / 365)) 
                if cogs and cogs != 0 and inventory is not None else None
            )
            
            efficiency_ratios.loc['DSO', year] = dso_val
            efficiency_ratios.loc['DPO', year] = dpo_val
            efficiency_ratios.loc['DIO', year] = dio_val
            efficiency_ratios.loc['Cash Conversion Cycle', year] = (
                dso_val + dio_val - dpo_val 
                if None not in [dso_val, dio_val, dpo_val] else None
            )
            
            # Calculate Leverage Ratios
            leverage_ratios.loc['Debt/Equity', year] = (
                total_debt / total_equity if total_equity and total_equity != 0 else None
            )
            leverage_ratios.loc['Debt/Assets', year] = (
                total_debt / total_assets if total_assets and total_assets != 0 else None
            )
            
        except Exception as e:
            if st.session_state.get('debug_mode', False):
                st.error(f"Error calculating ratios for {year}: {str(e)}")
            continue
    
    # Format the ratios
    def format_ratio(val, ratio_type: str) -> str:
        if pd.isna(val) or val is None:
            return 'N/A'
        if ratio_type == 'currency':
            return format_value(val)
        elif ratio_type == 'percentage':
            return f"{val * 100:.1f}%"
        elif ratio_type == 'days':
            return f"{val:.1f} days"
        else:  # decimal
            return f"{val:.2f}"
    
    # Apply formatting
    liquidity_ratios = liquidity_ratios.applymap(
        lambda x: format_ratio(x, 'decimal' if x != 'Working Capital' else 'currency')
    )
    profitability_ratios = profitability_ratios.applymap(
        lambda x: format_ratio(x, 'percentage')
    )
    efficiency_ratios = efficiency_ratios.applymap(
        lambda x: format_ratio(x, 'days')
    )
    leverage_ratios = leverage_ratios.applymap(
        lambda x: format_ratio(x, 'decimal')
    )
    
    return {
        'liquidity': liquidity_ratios,
        'profitability': profitability_ratios,
        'efficiency': efficiency_ratios,
        'leverage': leverage_ratios
    }

def display_financial_ratios(ticker: str, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame):
    """Display key financial ratios in a tabular format."""
    ratios = calculate_financial_ratios(income_stmt, balance_sheet, cash_flow)
    if ratios is None:
        st.warning(f"Could not calculate financial ratios for {ticker}. Insufficient data.")
        return
    
    # Display each ratio category
    st.markdown("### Liquidity Ratios")
    st.dataframe(ratios['liquidity'], use_container_width=True)
    st.markdown("""
    <div style="font-size: 0.85em; color: #666; margin-top: -15px; margin-bottom: 20px;">
        Current Ratio = Current Assets / Current Liabilities • 
        Quick Ratio = (Current Assets - Inventory) / Current Liabilities • 
        Working Capital = Current Assets - Current Liabilities
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Profitability Ratios")
    st.dataframe(ratios['profitability'], use_container_width=True)
    st.markdown("""
    <div style="font-size: 0.85em; color: #666; margin-top: -15px; margin-bottom: 20px;">
        Margins = Respective Income / Revenue • 
        ROA = Net Income / Total Assets • 
        ROE = Net Income / Total Equity
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Efficiency Ratios")
    st.dataframe(ratios['efficiency'], use_container_width=True)
    st.markdown("""
    <div style="font-size: 0.85em; color: #666; margin-top: -15px; margin-bottom: 20px;">
        DSO = Accounts Receivable / (Revenue/365) • 
        DPO = Accounts Payable / (COGS/365) • 
        DIO = Inventory / (COGS/365) • 
        Cash Conversion Cycle = DSO + DIO - DPO
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Leverage Ratios")
    st.dataframe(ratios['leverage'], use_container_width=True)
    st.markdown("""
    <div style="font-size: 0.85em; color: #666; margin-top: -15px; margin-bottom: 20px;">
        Debt/Equity = Total Debt / Total Equity • 
        Debt/Assets = Total Debt / Total Assets
    </div>
    """, unsafe_allow_html=True)
    
    # Add ratio insights analysis
    if ratios:
        # Get the most recent year available
        latest_year = next(iter(ratios['liquidity'].columns), None)
        if latest_year:
            display_ratio_insights(ratios, latest_year)

def display_ratio_insights(ratios: dict, year: str):
    """Display actionable insights for financial ratios using Streamlit components."""
    
    def fmt(value, is_percent=False):
        """Format values consistently."""
        if pd.isna(value) or value is None:
            return "N/A"
        if is_percent:
            return f"{value*100:.1f}%"
        return f"{value:.1f}"
    
    insights = []
    
    # Get the most recent year's ratios
    latest_ratios = {}
    for ratio_type, df in ratios.items():
        if year in df.columns:
            for ratio_name in df.index:
                try:
                    # Convert string values back to numbers if needed
                    val_str = df.loc[ratio_name, year]
                    if isinstance(val_str, str):
                        if '%' in val_str:
                            val = float(val_str.replace('%', '')) / 100
                        elif 'days' in val_str:
                            val = float(val_str.replace(' days', ''))
                        else:
                            val = float(val_str)
                    else:
                        val = val_str
                    latest_ratios[ratio_name] = val
                except:
                    continue
    
    if not latest_ratios:
        st.warning("No ratio data available for insights")
        return
    
    st.markdown("---")
    st.subheader("Ratio Insights & Analysis")
    st.caption(f"Interpretation of {year} financial ratios with industry benchmarks")
    
    # Create expandable sections for each category
    with st.expander("Liquidity Analysis", expanded=True):
        cols = st.columns(3)
        
        # Current Ratio
        cr = latest_ratios.get("Current Ratio")
        with cols[0]:
            st.metric("Current Ratio", 
                     fmt(cr) if cr is not None else "N/A",
                     help="Current assets / current liabilities")
            if cr is not None:
                if cr < 1:
                    st.error("🔴 <1: May struggle to meet obligations (Ideal range: 1.5-3)")
                elif cr > 3:
                    st.warning("🟡 >3: Possible excess idle assets")
                else:
                    st.success("🟢 Healthy (1-3)")
        
        # Quick Ratio
        qr = latest_ratios.get("Quick Ratio")
        with cols[1]:
            st.metric("Quick Ratio", 
                     fmt(qr) if qr is not None else "N/A",
                     help="(Current assets - inventory) / current liabilities")
            if qr is not None:
                if qr < 1:
                    st.error("🔴 <1: Liquidity concern without inventory")
                elif qr < 2:
                    st.warning("🟡 Moderate (1-2)")
                else:
                    st.success("🟢 Strong (>2)")
        
        # Working Capital
        wc = latest_ratios.get("Working Capital")
        with cols[2]:
            st.metric("Working Capital", 
                     fmt(wc) if wc is not None else "N/A",
                     help="Current assets - current liabilities")
            if wc is not None:
                if wc < 0:
                    st.error("🔴 Negative: Liquidity risk")
                else:
                    st.success("🟢 Positive: Solvency OK")
    
    with st.expander("Profitability Analysis", expanded=True):
        cols = st.columns(5)
        
        # Gross Margin
        gm = latest_ratios.get("Gross Margin")
        with cols[0]:
            st.metric("Gross Margin", 
                     fmt(gm, True) if gm is not None else "N/A",
                     help="Gross profit / revenue")
            if gm is not None:
                if gm < 0.2:
                    st.error("🔴 <20%: High costs (Target: >40%)")
                elif gm < 0.4:
                    st.warning("🟡 Moderate (20-40%)")
                else:
                    st.success("🟢 Strong (>40%)")
        
        # Operating Margin
        om = latest_ratios.get("Operating Margin")
        with cols[1]:
            st.metric("Operating Margin", 
                     fmt(om, True) if om is not None else "N/A",
                     help="Operating income / revenue")
            if om is not None:
                if om < 0.1:
                    st.error("🔴 <10%: Heavy ops costs (Target: >20%)")
                elif om < 0.2:
                    st.warning("🟡 Moderate (10-20%)")
                else:
                    st.success("🟢 Healthy (>20%)")
        
        # Net Margin
        nm = latest_ratios.get("Net Margin")
        with cols[2]:
            st.metric("Net Margin", 
                     fmt(nm, True) if nm is not None else "N/A",
                     help="Net income / revenue")
            if nm is not None:
                if nm < 0:
                    st.error("🔴 Negative: Unprofitable (Target: >20%)")
                elif nm > 0.2:
                    st.success("🟢 Excellent (>20%)")
                else:
                    st.warning("🟡 Moderate (0-20%)")
        
        # ROA
        roa = latest_ratios.get("ROA")
        with cols[3]:
            st.metric("ROA", 
                     fmt(roa, True) if roa is not None else "N/A",
                     help="Net income / total assets")
            if roa is not None:
                if roa < 0:
                    st.error("🔴 Negative: Asset misuse (Target: >10%)")
                elif roa > 0.1:
                    st.success("🟢 Strong (>10%)")
                else:
                    st.warning("🟡 Average (0-10%)")
        
        # ROE
        roe = latest_ratios.get("ROE")
        with cols[4]:
            st.metric("ROE", 
                     fmt(roe, True) if roe is not None else "N/A",
                     help="Net income / total equity")
            if roe is not None:
                if roe < 0:
                    st.error("🔴 Negative: Equity erosion (Target: >15%)")
                elif roe > 0.15:
                    st.success("🟢 High (>15%)")
                else:
                    st.warning("🟡 Average (0-15%)")
    
    with st.expander("Efficiency Analysis", expanded=True):
        cols = st.columns(4)
        
        # DSO
        dso = latest_ratios.get("DSO")
        with cols[0]:
            st.metric("DSO", 
                     fmt(dso) if dso is not None else "N/A",
                     help="Days Sales Outstanding")
            if dso is not None:
                if dso > 60:
                    st.error("🔴 >60: Slow collections (Target: <30)")
                elif dso < 30:
                    st.success("🟢 <30: Fast payments")
                else:
                    st.warning("🟡 Average (30-60)")
        
        # DPO
        dpo = latest_ratios.get("DPO")
        with cols[1]:
            st.metric("DPO", 
                     fmt(dpo) if dpo is not None else "N/A",
                     help="Days Payable Outstanding")
            if dpo is not None:
                if dpo > 90:
                    st.success("🟢 >90: Good credit terms")
                elif dpo < 30:
                    st.error("🔴 <30: Short payment terms (Target: 30-90)")
                else:
                    st.warning("🟡 Normal (30-90)")
        
        # DIO
        dio = latest_ratios.get("DIO")
        with cols[2]:
            st.metric("DIO", 
                     fmt(dio) if dio is not None else "N/A",
                     help="Days Inventory Outstanding")
            if dio is not None:
                if dio > 90:
                    st.error("🔴 >90: Slow turnover (Target: <30)")
                elif dio < 30:
                    st.success("🟢 <30: Fast turnover")
                else:
                    st.warning("🟡 Average (30-90)")
        
        # CCC
        ccc = latest_ratios.get("Cash Conversion Cycle")
        with cols[3]:
            st.metric("CCC", 
                     fmt(ccc) if ccc is not None else "N/A",
                     help="Cash Conversion Cycle")
            if ccc is not None:
                if ccc < 0:
                    st.success("🟢 Negative: Very efficient")
                elif ccc < 60:
                    st.warning("🟡 Reasonable (<60)")
                else:
                    st.error("🔴 >60: Slow conversion (Target: <60)")
    
    # Leverage Analysis
    with st.expander("🏗️ Leverage Analysis", expanded=True):
        cols = st.columns(2)
        
        # Debt/Equity
        de = latest_ratios.get("Debt/Equity")
        with cols[0]:
            st.metric("Debt/Equity", 
                     fmt(de) if de is not None else "N/A",
                     help="Total debt / total equity")
            if de is not None:
                if de > 2:
                    st.error("🔴 >2: High leverage (Target: <1)")
                elif de > 1:
                    st.warning("🟡 Moderate (1-2)")
                else:
                    st.success("🟢 Conservative (<1)")
        
        # Debt/Assets
        da = latest_ratios.get("Debt/Assets")
        with cols[1]:
            st.metric("Debt/Assets", 
                     fmt(da) if da is not None else "N/A",
                     help="Total debt / total assets")
            if da is not None:
                if da > 0.5:
                    st.error("🔴 >0.5: Asset-heavy debt (Target: <0.3)")
                elif da > 0.3:
                    st.warning("🟡 Moderate (0.3-0.5)")
                else:
                    st.success("🟢 Conservative (<0.3)")
    
    # Industry Comparison Section
    st.markdown("---")
    st.markdown("#### 🏭 Industry Benchmark Comparison")
    st.warning("Industry benchmark data coming soon! We'll soon add comparison against sector averages.")

def monte_carlo_simulation(start_value: float, mean_return: float, volatility: float, 
                          years: int = 10, simulations: int = 500, steps_per_year: int = 252) -> np.ndarray:
    """Run Monte Carlo simulation of portfolio growth using geometric Brownian motion.
    
    Args:
        start_value: Initial portfolio value
        mean_return: Expected annual return (decimal)
        volatility: Annual volatility (standard deviation)
        years: Time horizon in years
        simulations: Number of simulation paths
        steps_per_year: Number of time steps per year
        
    Returns:
        Numpy array of simulation paths (simulations x time steps)
    """
    dt = 1 / steps_per_year
    total_steps = int(steps_per_year * years)
    results = np.zeros((simulations, total_steps))
    
    for i in range(simulations):
        prices = [start_value]
        for _ in range(total_steps - 1):
            shock = np.random.normal(loc=(mean_return - 0.5 * volatility**2) * dt, 
                                    scale=volatility * np.sqrt(dt))
            prices.append(prices[-1] * np.exp(shock))
        results[i] = prices
        
    return results

def calculate_annual_returns(close_prices: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Calculate annual returns for the portfolio"""
    # Resample to yearly closing prices
    yearly_prices = close_prices.resample('Y').last()
    # Calculate yearly returns
    yearly_returns = yearly_prices.pct_change().dropna()
    # Calculate weighted portfolio returns
    port_returns = (yearly_returns * weights).sum(axis=1)
    return port_returns

def fetch_stock_info_with_retry(ticker: str, max_retries: int = 3, delay: float = 1.0) -> Optional[Dict]:
    """Fetch stock info with proxy rotation and retry logic."""
    for attempt in range(max_retries):
        proxy = get_proxy_dict()
        try:
            if proxy:
                yf.set_config(proxy=proxy)
            else:
                yf.set_config(proxy=None)
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            if info and "quoteType" in info:
                return info
            # Fallback to fast_info
            fast_info = getattr(ticker_obj, "fast_info", None)
            if fast_info:
                return fast_info
        except Exception as e:
            if st.session_state.get('debug_mode', False):
                st.warning(f"Attempt {attempt+1} failed for {ticker}: {e}")
        time.sleep(delay * (2 ** attempt))  # Exponential backoff
    return None

def fetch_all_stock_info(tickers: List[str], max_workers: int = 5) -> Dict[str, Optional[Dict]]:
    """Fetch info for all tickers in parallel with retry and proxy rotation."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(fetch_stock_info_with_retry, ticker): ticker for ticker in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except Exception as exc:
                results[ticker] = None
    return results

def get_cached_stock_info(ticker: str) -> Optional[Dict]:
    """Get stock info from session cache or fetch if not present."""
    cache = st.session_state.setdefault('stock_info_cache', {})
    if ticker in cache:
        return cache[ticker]
    info = fetch_stock_info_with_retry(ticker)
    cache[ticker] = info
    return info

def get_proxy_dict_enhanced(probability=0.3, max_retries=2):
    """Improved proxy fetcher with fallback"""
    if not PROXY_AVAILABLE or random.random() > probability:
        return None
    for _ in range(max_retries):
        try:
            proxy = FreeProxy(rand=True, timeout=5).get()
            if proxy:
                return {"http": proxy, "https": proxy}
        except Exception:
            continue
    return None

# Enhanced stock history fetcher with retry, delay, and proxy fallback
def fetch_stock_history_enhanced(
    ticker,
    period=None,
    interval="1d",
    start=None,
    end=None,
    max_retries=3
):
    """Improved stock history fetcher with retry logic"""
    for attempt in range(max_retries):
        try:
            proxy = get_proxy_dict_enhanced(probability=0.3)
            if proxy:
                yf.set_config(proxy=proxy)
            else:
                yf.set_config(proxy=None)
            ticker_obj = yf.Ticker(ticker)
            time.sleep(0.5 + random.random())  # Add small delay between requests
            if start and end:
                hist = ticker_obj.history(
                    start=start.strftime('%Y-%m-%d') if isinstance(start, datetime) else start,
                    end=end.strftime('%Y-%m-%d') if isinstance(end, datetime) else end,
                    interval=interval,
                    timeout=10
                )
            elif period:
                hist = ticker_obj.history(
                    period=period,
                    interval=interval,
                    timeout=10
                )
            else:
                return None
            if hist is not None and not hist.empty:
                return hist
        except Exception as e:
            if attempt == max_retries - 1:
                if st.session_state.get('debug_mode', False):
                    st.error(f"Final attempt failed for {ticker}: {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff
    return None

# Caching strategy improvements for stock info
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_info_cached(ticker):
    """Cached version with better error handling"""
    try:
        info = fetch_stock_info_with_retry(ticker)
        if not info or "quoteType" not in info:
            return None
        return info
    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Cache error for {ticker}: {str(e)}")
        return None

# Parallel fetching for multiple stocks with rate control
def fetch_multiple_stocks_parallel(tickers, max_workers=4):
    """Fetch multiple stocks with controlled parallelism"""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(fetch_stock_history_enhanced, ticker): ticker 
            for ticker in tickers
        }
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except Exception as exc:
                results[ticker] = None
                if st.session_state.get('debug_mode', False):
                    st.warning(f"{ticker} generated an exception: {exc}")
    return results

# Main App Content
def main():
    """Main application function"""
    st.title("Optimal Portfolio Dashboard")
    
    # Check for missing dependencies
    missing_deps = []
    if not YFINANCE_AVAILABLE:
        missing_deps.append("yfinance")
    if not WEB_SCRAPING_AVAILABLE:
        missing_deps.append("requests, beautifulsoup4")
    if not SCIPY_AVAILABLE:
        missing_deps.append("scipy")
    
    if missing_deps:
        st.error(f"Missing dependencies: {', '.join(missing_deps)}")
        st.info("Please install missing packages using: pip install " + " ".join(missing_deps))
        st.stop()
    
    # Debug mode toggle
    st.session_state['debug_mode'] = st.sidebar.checkbox("Debug Mode", value=st.session_state.get('debug_mode', False))
    
    # Portfolio input section
    st.subheader("Portfolio Setup")
    
    # Fetch S&P 500 tickers
    with st.spinner("Loading S&P 500 stocks..."):
        sp500_tickers = fetch_sp500_tickers()
    
    # Stock selection interface - use full width
    selected_from_dropdown = st.multiselect(
        "Select stocks from S&P 500:",
        options=sp500_tickers if sp500_tickers else [],
        default=['AAPL', 'GOOGL', 'MSFT', 'TSLA'] if sp500_tickers and all(t in sp500_tickers for t in ['AAPL', 'GOOGL', 'MSFT', 'TSLA']) else (sp500_tickers[:4] if sp500_tickers else []),
        help="Select multiple stocks from the S&P 500"
    )
    
    # Combine selections and remove duplicates
    final_tickers = remove_duplicates(selected_from_dropdown)
    
    # Risk level selector with radio buttons
    risk_levels = ["Low", "Moderate", "High", "Custom"]
    current_risk_level = st.session_state.get('risk_level', 'Moderate')
    
    # Display risk level descriptions
    risk_descriptions = {
        "Low": "🛡️ Minimize portfolio volatility for conservative investors",
        "Moderate": "⚖️ Maximize Sharpe ratio for balanced risk-return",
        "High": "🚀 Maximize expected returns for aggressive investors",
        "Custom": "🎛️ Manually set your own portfolio weights"
    }
    
    # Create radio buttons for risk level selection
    selected_risk_level = st.radio(
        "Select Risk Level:",
        options=risk_levels,
        index=risk_levels.index(current_risk_level) if current_risk_level in risk_levels else 1,
        help="Choose your investment strategy",
        format_func=lambda x: f"{x} - {risk_descriptions[x]}"
    )
    
    # Update session state and handle custom weights
    if selected_risk_level != st.session_state.get('risk_level'):
        st.session_state['risk_level'] = selected_risk_level
        # Reset custom weights when changing risk level
        if selected_risk_level != "Custom":
            st.session_state['custom_weights'] = {}
        if st.session_state.get('portfolio_created', False):
            st.rerun()
    
    # Handle custom weights immediately after risk level selection
    if selected_risk_level == "Custom" and final_tickers:
        st.markdown("### 🎛️ Custom Weight Allocation")
        
        # Initialize or get existing custom weights
        custom_weights = st.session_state.get('custom_weights', {})
        
        # Update custom weights for current tickers
        for ticker in final_tickers:
            if ticker not in custom_weights:
                # Initialize new tickers with equal weight
                custom_weights[ticker] = 100.0/len(final_tickers)
        
        # Remove weights for tickers that are no longer selected
        custom_weights = {k: v for k, v in custom_weights.items() if k in final_tickers}
        
        # Normalize weights if we have tickers
        if final_tickers:
            total_weight = sum(custom_weights.values())
            if total_weight > 0:
                custom_weights = {k: (v/total_weight)*100 for k, v in custom_weights.items()}
            else:
                custom_weights = {ticker: 100.0/len(final_tickers) for ticker in final_tickers}
        
        total_weight = 0.0
        cols = st.columns(min(len(final_tickers), 4))  # Max 4 columns for better layout
        
        for i, ticker in enumerate(final_tickers):
            with cols[i % len(cols)]:
                current_weight = custom_weights.get(ticker, 100.0/len(final_tickers))
                weight = st.number_input(
                    f"{ticker} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=current_weight,
                    step=1.0,
                    format="%.1f",
                    key=f"weight_{ticker}"
                )
                custom_weights[ticker] = weight
                total_weight += weight
        
        # Update session state
        st.session_state['custom_weights'] = custom_weights
        
        # Validate total weight
        if abs(total_weight - 100.0) > 0.01:
            st.warning(f"⚠️ Total weights must sum to 100%. Current total: {total_weight:.1f}%")
            st.stop()
        else:
            st.success("✅ Weights sum to 100%")
    
    # --- Year Range Selection ---
    current_year = datetime.today().year
    year_options = list(range(2010, current_year + 1))
    
    # Get stored years or use defaults
    stored_start_year = st.session_state.get('start_year', 2018)
    stored_end_year = st.session_state.get('end_year', current_year)
    
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox(
            "Start Year",
            options=year_options,
            index=year_options.index(stored_start_year) if stored_start_year in year_options else year_options.index(2018)
        )
    with col2:
        end_year = st.selectbox(
            "End Year",
            options=year_options,
            index=year_options.index(stored_end_year) if stored_end_year in year_options else year_options.index(current_year)
        )

    # Ensure end year is after start year
    if start_year >= end_year:
        st.warning("End year must be after start year.")
        st.stop()

    # Create date range from years
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    # Store years in session state
    st.session_state['start_year'] = start_year
    st.session_state['end_year'] = end_year
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date
    
    # Create portfolio button - centered
    if final_tickers:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Create Optimal Portfolio", use_container_width=True):
                st.session_state['portfolio_tickers'] = final_tickers
                st.session_state['portfolio_created'] = True
                st.success(f"Portfolio created with {len(final_tickers)} stocks!")
    
    # Show portfolio analysis if created
    if st.session_state.get('portfolio_created', False):
        tickers = st.session_state.get('portfolio_tickers', [])
        
        st.subheader("Portfolio Analysis")
        
        # Display optimal portfolio with current risk level
        display_optimal_portfolio(tickers)
        
        # Display benchmark comparison
        display_benchmark_comparison(tickers)
        
        # Chart section with timeframe selector
        st.subheader("Portfolio Performance Chart")
        
        # Timeframe selector for chart only
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            chart_timeframe = st.selectbox(
                "Select Chart Timeframe:",
                options=['1D', '5D', '1M', '6M', '1Y', 'YTD', '5Y', '10Y'],
                index=2,  # Default to 1M
                help="Choose the time period for the chart display",
                key="chart_timeframe"
            )
        
        # Create and display chart using selected timeframe
        create_portfolio_chart(tickers, chart_timeframe)

# Gradient-based color formatting for DataFrame

def gradient_performance(val, positive_is_good=True):
    """Apply conditional formatting with red-yellow-green gradient"""
    if isinstance(val, str) and '%' in val:
        try:
            num = float(val.replace('%', '').replace('+', ''))
            if positive_is_good:
                # For metrics where higher is better (returns, change %)
                norm = min(max((num + 100) / 200, 0), 1)
                color = matplotlib.colors.rgb2hex(matplotlib.cm.RdYlGn(norm))
            else:
                # For metrics where lower is better (volatility)
                norm = min(max(num / 100, 0), 1)
                color = matplotlib.colors.rgb2hex(matplotlib.cm.RdYlGn_r(norm))
            text_color = 'white' if norm > 0.7 or norm < 0.3 else 'black'
            return f'background-color: {color}; color: {text_color}'
        except:
            pass
    return ''

if __name__ == "__main__":
    main()