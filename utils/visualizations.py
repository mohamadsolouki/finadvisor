"""
Visualization utilities for creating charts and graphs
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def create_trend_chart(data, title, metric_name):
    """Create an interactive trend chart"""
    years = [col for col in data.columns if col not in ['Parameters', 'Currency']]
    values = []
    
    for year in years:
        val = str(data[year].iloc[0]).replace(',', '')
        try:
            values.append(float(val))
        except:
            values.append(0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years,
        y=values,
        mode='lines+markers',
        name=metric_name,
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Value',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_comparison_chart(data, title):
    """Create a comparison chart for multiple metrics"""
    years = [col for col in data.columns if col not in ['Parameters', 'Currency']]
    
    fig = go.Figure()
    
    for idx, row in data.iterrows():
        values = []
        for year in years:
            val = str(row[year]).replace(',', '')
            try:
                values.append(float(val))
            except:
                values.append(0)
        
        fig.add_trace(go.Scatter(
            x=years,
            y=values,
            mode='lines+markers',
            name=row['Parameters']
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Value',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    return fig


def create_candlestick_chart(hist, ticker, ma_periods=[20, 50, 200]):
    """Create candlestick chart with moving averages and Bollinger Bands"""
    # Calculate moving averages
    for period in ma_periods:
        hist[f'MA{period}'] = hist['Close'].rolling(window=period).mean()
    
    # Calculate Bollinger Bands
    hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
    hist['BB_Std'] = hist['Close'].rolling(window=20).std()
    hist['BB_Upper'] = hist['BB_Middle'] + (hist['BB_Std'] * 2)
    hist['BB_Lower'] = hist['BB_Middle'] - (hist['BB_Std'] * 2)
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name='OHLC',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # Moving averages
    colors = ['#ffa726', '#42a5f5', '#ef5350']
    for period, color in zip(ma_periods, colors):
        fig.add_trace(go.Scatter(
            x=hist.index, 
            y=hist[f'MA{period}'], 
            name=f'{period}-Day MA',
            line=dict(color=color, width=1.5 if period != 200 else 2)
        ))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist['BB_Upper'], 
        name='BB Upper',
        line=dict(color='gray', width=1, dash='dot'),
        showlegend=False, opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist['BB_Lower'], 
        name='BB Lower',
        line=dict(color='gray', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
        showlegend=False, opacity=0.5
    ))
    
    fig.update_layout(
        title=f'{ticker} Stock Price with Technical Indicators',
        xaxis_title='Date', 
        yaxis_title='Price (USD)',
        hovermode='x unified', 
        template='plotly_white', 
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    return fig, hist


def create_rsi_chart(hist):
    """Create RSI (Relative Strength Index) chart"""
    # Calculate RSI
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist['RSI'], 
        name='RSI', 
        line=dict(color='purple', width=2)
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                  annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", 
                  annotation_text="Oversold (30)")
    fig.update_layout(
        yaxis_title='RSI', 
        xaxis_title='Date',
        template='plotly_white', 
        height=300
    )
    
    return fig, hist


def create_volume_chart(hist):
    """Create volume bar chart"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hist.index, 
        y=hist['Volume'], 
        name='Volume',
        marker_color='lightblue'
    ))
    fig.update_layout(
        xaxis_title='Date', 
        yaxis_title='Volume',
        template='plotly_white', 
        height=400
    )
    return fig


def create_returns_distribution(hist):
    """Create returns distribution histogram"""
    returns = hist['Close'].pct_change().dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns * 100, 
        nbinsx=50, 
        name='Daily Returns',
        marker_color='#3498db'
    ))
    
    # Add VaR line
    var_95 = returns.quantile(0.05) * 100
    fig.add_vline(x=var_95, line_dash="dash", line_color="red",
                  annotation_text="VaR 95%", annotation_position="top")
    
    fig.update_layout(
        xaxis_title='Daily Return (%)', 
        yaxis_title='Frequency',
        template='plotly_white', 
        height=400, 
        showlegend=False
    )
    return fig


def create_drawdown_chart(hist):
    """Create drawdown chart"""
    returns = hist['Close'].pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max - 1) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown,
        fill='tozeroy', name='Drawdown',
        line=dict(color='#c0392b', width=2),
        fillcolor='rgba(192, 57, 43, 0.3)'
    ))
    fig.update_layout(
        xaxis_title='Date', 
        yaxis_title='Drawdown (%)',
        template='plotly_white', 
        height=400
    )
    return fig


def create_rolling_volatility_chart(hist, window=30):
    """Create rolling volatility chart"""
    returns = hist['Close'].pct_change().dropna()
    rolling_vol = returns.rolling(window=window).std() * (252 ** 0.5) * 100
    avg_vol = rolling_vol.mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_vol.index, y=rolling_vol,
        name='Rolling Volatility',
        line=dict(color='#e74c3c', width=2),
        fill='tozeroy'
    ))
    fig.add_hline(y=avg_vol, line_dash="dash", line_color="gray", 
                  annotation_text=f"Avg: {avg_vol:.2f}%")
    fig.update_layout(
        xaxis_title='Date', 
        yaxis_title='Volatility (%)',
        template='plotly_white', 
        height=400
    )
    return fig
