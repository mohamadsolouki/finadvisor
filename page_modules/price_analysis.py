"""
Price Analysis Page Module
Displays stock price charts, technical indicators, and trading analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_fetcher import get_historical_data, get_stock_info
from utils.visualizations import create_candlestick_chart, create_rsi_chart


def display_price_analysis(ticker, cached_info=None):
    """Display price analysis page"""
    st.subheader("📈 Stock Price Analysis & Technical Indicators")
    
    # Time period selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### Market Data")
    with col2:
        period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=5)
    with col3:
        st.write("")  # Spacer
    
    # Use cached info if available, otherwise fetch
    hist = get_historical_data(ticker, period=period)
    info = cached_info if cached_info else get_stock_info(ticker)
    
    if hist is None or hist.empty:
        st.warning("Unable to fetch stock price data. Please try again later or check your internet connection.")
        st.info("Yahoo Finance may be rate-limiting requests. Data is cached for 2 hours once successfully loaded.")
        return
    
    # Current price metrics
    st.markdown("### 💰 Current Price Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_price = hist['Close'].iloc[-1]
    prev_close = info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price)
    day_change = current_price - prev_close
    day_change_pct = (day_change / prev_close) * 100 if prev_close != 0 else 0
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{day_change_pct:+.2f}%")
    
    with col2:
        high_52w = info.get('fiftyTwoWeekHigh', hist['High'].max())
        distance_from_high = ((current_price - high_52w) / high_52w) * 100
        st.metric("52-Week High", f"${high_52w:.2f}", f"{distance_from_high:+.1f}%")
    
    with col3:
        low_52w = info.get('fiftyTwoWeekLow', hist['Low'].min())
        distance_from_low = ((current_price - low_52w) / low_52w) * 100
        st.metric("52-Week Low", f"${low_52w:.2f}", f"{distance_from_low:+.1f}%")
    
    with col4:
        avg_volume = hist['Volume'].tail(30).mean()
        current_volume = hist['Volume'].iloc[-1]
        volume_change = ((current_volume - avg_volume) / avg_volume) * 100
        st.metric("Avg Volume (30d)", f"{avg_volume/1e6:.2f}M", f"{volume_change:+.1f}%")
    
    with col5:
        market_cap = info.get('marketCap', 0) / 1e9
        st.metric("Market Cap", f"${market_cap:.2f}B")
    
    st.markdown("---")
    
    # Price chart with technical indicators
    st.markdown("### 📊 Price History with Technical Indicators")
    
    fig, hist = create_candlestick_chart(hist, ticker)
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI Chart
        st.markdown("#### Relative Strength Index (RSI)")
        fig_rsi, hist = create_rsi_chart(hist)
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        current_rsi = hist['RSI'].iloc[-1]
        if current_rsi > 70:
            st.warning(f"⚠️ RSI: {current_rsi:.1f} - Stock may be overbought")
        elif current_rsi < 30:
            st.success(f"✅ RSI: {current_rsi:.1f} - Stock may be oversold")
        else:
            st.info(f"ℹ️ RSI: {current_rsi:.1f} - Stock is in neutral territory")
    
    with col2:
        # MACD would go here (simplified version)
        st.markdown("#### Price Position vs Moving Averages")
        
        ma_status = []
        if not pd.isna(hist['MA20'].iloc[-1]):
            ma_status.append(("20-Day MA", current_price > hist['MA20'].iloc[-1]))
        if not pd.isna(hist['MA50'].iloc[-1]):
            ma_status.append(("50-Day MA", current_price > hist['MA50'].iloc[-1]))
        if not pd.isna(hist['MA200'].iloc[-1]):
            ma_status.append(("200-Day MA", current_price > hist['MA200'].iloc[-1]))
        
        for ma_name, is_above in ma_status:
            status = "✅ Above" if is_above else "❌ Below"
            st.write(f"{status} {ma_name}")
        
        st.markdown("---")
        
        # Bollinger Band position
        if not pd.isna(hist['BB_Upper'].iloc[-1]):
            bb_position = (current_price - hist['BB_Lower'].iloc[-1]) / (hist['BB_Upper'].iloc[-1] - hist['BB_Lower'].iloc[-1])
            st.metric("Bollinger Band Position", f"{bb_position*100:.1f}%")
            
            if bb_position > 1:
                st.warning("⚠️ Price above upper band")
            elif bb_position < 0:
                st.success("✅ Price below lower band")
            else:
                st.info("ℹ️ Price within bands")
    
    st.markdown("---")
    
    # Volume analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trading Volume")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume',
                                 marker_color='lightblue'))
        fig_vol.update_layout(xaxis_title='Date', yaxis_title='Volume',
                             template='plotly_white', height=400)
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col2:
        st.subheader("Price Returns Distribution")
        returns = hist['Close'].pct_change().dropna()
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=returns, nbinsx=50, name='Daily Returns',
                                        marker_color='lightgreen'))
        fig_dist.update_layout(xaxis_title='Daily Return', yaxis_title='Frequency',
                              template='plotly_white', height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Performance metrics
    st.subheader("Performance Metrics")
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    returns_1m = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-21]) - 1) * 100 if len(hist) > 21 else 0
    returns_3m = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-63]) - 1) * 100 if len(hist) > 63 else 0
    returns_1y = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-252]) - 1) * 100 if len(hist) > 252 else 0
    returns_ytd = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
    
    with perf_col1:
        st.metric("1 Month Return", f"{returns_1m:+.2f}%")
    with perf_col2:
        st.metric("3 Month Return", f"{returns_3m:+.2f}%")
    with perf_col3:
        st.metric("1 Year Return", f"{returns_1y:+.2f}%")
    with perf_col4:
        st.metric("YTD Return", f"{returns_ytd:+.2f}%")
