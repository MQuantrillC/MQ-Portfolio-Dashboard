import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_pie_chart(allocation):
    """
    Create a pie chart for portfolio allocation
    """
    if not allocation:
        return None
    
    fig = go.Figure(data=[go.Pie(
        labels=list(allocation.keys()),
        values=list(allocation.values()),
        hole=.3
    )])
    
    fig.update_layout(
        title="Portfolio Allocation",
        showlegend=True
    )
    
    return fig

def create_performance_chart(portfolio_values):
    """
    Create a line chart for portfolio performance
    """
    if portfolio_values is None or portfolio_values.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=portfolio_values.index,
        y=portfolio_values['Total'],
        mode='lines',
        name='Portfolio Value'
    ))
    
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        showlegend=True
    )
    
    return fig

def create_returns_heatmap(returns):
    """
    Create a heatmap of monthly returns
    """
    if returns is None or returns.empty:
        return None
    
    # Calculate monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create a pivot table for the heatmap
    monthly_returns_pivot = monthly_returns.pivot_table(
        index=monthly_returns.index.year,
        columns=monthly_returns.index.month,
        values=monthly_returns
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=monthly_returns_pivot.values,
        x=monthly_returns_pivot.columns,
        y=monthly_returns_pivot.index,
        colorscale='RdYlGn',
        zmid=0
    ))
    
    fig.update_layout(
        title="Monthly Returns Heatmap",
        xaxis_title="Month",
        yaxis_title="Year"
    )
    
    return fig 