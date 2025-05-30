import pandas as pd
import numpy as np
from data_loader import get_stock_data

class Portfolio:
    def __init__(self, holdings):
        """
        Initialize portfolio with holdings
        holdings: dict with ticker as key and number of shares as value
        """
        self.holdings = holdings
        self.data = None
        self.returns = None
        
    def fetch_data(self, period="1y"):
        """Fetch historical data for all holdings"""
        self.data = get_stock_data(list(self.holdings.keys()), period)
        if self.data is not None:
            self._calculate_returns()
    
    def _calculate_returns(self):
        """Calculate daily returns for the portfolio"""
        if self.data is None:
            self.returns = None
            return

        # Ensure index is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data.index):
            self.data.index = pd.to_datetime(self.data.index)

        # Handle both single and multiple tickers
        if isinstance(self.data.columns, pd.MultiIndex):
            close_prices = pd.DataFrame({
                ticker: self.data['Close', ticker]
                for ticker in self.holdings.keys()
                if ('Close', ticker) in self.data.columns
            })
        else:
            ticker = list(self.holdings.keys())[0]
            close_prices = pd.DataFrame({ticker: self.data['Close']})

        # Drop columns with all NaNs
        close_prices = close_prices.dropna(axis=1, how='all')
        # Forward-fill and back-fill to handle missing data
        close_prices = close_prices.ffill().bfill()

        # Calculate returns
        if not close_prices.empty:
            returns = close_prices.pct_change().dropna(how='all')
            if not returns.empty:
                self.returns = returns
            else:
                self.returns = None
        else:
            self.returns = None
    
    def get_allocation(self):
        """Calculate current portfolio allocation"""
        if self.data is None:
            return None
        
        allocation = {}
        total_value = 0
        if isinstance(self.data.columns, pd.MultiIndex):
            # Multiple tickers
            current_prices = {ticker: self.data['Close', ticker].iloc[-1] for ticker in self.holdings.keys() if ('Close', ticker) in self.data.columns}
        else:
            # Single ticker
            ticker = list(self.holdings.keys())[0]
            if 'Close' in self.data.columns:
                current_prices = {ticker: self.data['Close'].iloc[-1]}
            else:
                current_prices = {}
        
        for ticker, shares in self.holdings.items():
            if ticker in current_prices:
                value = shares * current_prices[ticker]
                allocation[ticker] = value
                total_value += value
        
        # Convert to percentages
        if total_value > 0:
            allocation = {k: (v/total_value)*100 for k, v in allocation.items()}
        
        return allocation
    
    def get_metrics(self):
        """Calculate portfolio metrics"""
        if self.returns is None:
            return None
        
        metrics = {
            'Total Return': (self.returns + 1).prod() - 1,
            'Annualized Return': (1 + self.returns.mean()) ** 252 - 1,
            'Volatility': self.returns.std() * np.sqrt(252),
            'Sharpe Ratio': (self.returns.mean() / self.returns.std()) * np.sqrt(252)
        }
        
        return metrics 