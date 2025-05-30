# Portfolio Dashboard

A Streamlit-based dashboard for tracking and analyzing your investment portfolio.

## Features

- Add and track multiple stocks
- View current portfolio allocation
- Monitor portfolio performance over time
- Analyze key portfolio metrics
- Visualize monthly returns with a heatmap

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run app.py
```

## Usage

1. Enter stock tickers and number of shares in the sidebar
2. View your portfolio allocation in the pie chart
3. Check key metrics like total return and volatility
4. Monitor performance over time with the line chart
5. Analyze monthly returns with the heatmap

## Note

This dashboard uses Yahoo Finance data through the yfinance package. Stock prices and data may be delayed by 15-20 minutes. 