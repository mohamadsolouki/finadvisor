"""
Price Analysis Page Module
Displays stock price charts, technical indicators, and trading analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data_fetcher import get_historical_data, get_stock_info
from utils.visualizations import create_candlestick_chart, create_rsi_chart


def display_price_analysis(ticker, cached_info=None):
    """Display price analysis page"""
    st.subheader("📈 Stock Price Analysis & Technical Indicators")
    
    # Date range selector
    st.markdown("### 📅 Analysis Period")
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=pd.to_datetime("2020-01-01"),
            key="price_start_date"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=pd.to_datetime("2024-12-31"),
            key="price_end_date"
        )
    
    # Validate dates
    if start_date >= end_date:
        st.error("⚠️ Start date must be before end date")
        return
    
    st.markdown("### Market Data")
    
    # Use cached info if available, otherwise fetch
    hist = get_historical_data(ticker, start_date=start_date, end_date=end_date)
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
    
    returns_1m = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-21]) - 1) * 100 if len(hist) > 21 else None
    returns_3m = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-63]) - 1) * 100 if len(hist) > 63 else None
    returns_1y = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-252]) - 1) * 100 if len(hist) > 252 else None
    
    # Calculate YTD return - find first trading day of current year
    current_year = pd.Timestamp.now().year
    ytd_data = hist[hist.index.year == current_year]
    if len(ytd_data) > 0:
        returns_ytd = ((hist['Close'].iloc[-1] / ytd_data['Close'].iloc[0]) - 1) * 100
    else:
        returns_ytd = None
    
    with perf_col1:
        st.metric("1 Month Return", f"{returns_1m:+.2f}%" if returns_1m is not None else "N/A")
    with perf_col2:
        st.metric("3 Month Return", f"{returns_3m:+.2f}%" if returns_3m is not None else "N/A")
    with perf_col3:
        st.metric("1 Year Return", f"{returns_1y:+.2f}%" if returns_1y is not None else "N/A")
    with perf_col4:
        st.metric("YTD Return", f"{returns_ytd:+.2f}%" if returns_ytd is not None else "N/A")
    
    st.markdown("---")
    
    # AI-Powered Price Analysis
    st.markdown("### 🤖 AI-Powered Price & Technical Analysis")
    st.markdown("Comprehensive interpretation of price trends, technical indicators, and trading patterns")
    
    # Import AI insights generator
    from utils.ai_insights_generator import AIInsightsGenerator
    from pathlib import Path
    
    # Initialize AI generator
    data_dir = Path("data")
    ai_generator = AIInsightsGenerator(data_dir)
    
    if ai_generator.enabled:
        with st.spinner("🧠 Generating comprehensive price analysis..."):
            # Get company name from info
            company_name = info.get('longName', info.get('shortName', ticker))
            
            # Use the same hist data that's being displayed in charts
            # Filter data from 2020 onwards
            hist_from_2020 = hist[hist.index >= '2020-01-01']
            
            # Get 2020 baseline
            start_2020 = None
            price_2020 = None
            if not hist_from_2020.empty:
                start_2020 = hist_from_2020.index[0]
                price_2020 = hist_from_2020['Close'].iloc[0]
            else:
                # If no 2020 data, use earliest available
                start_2020 = hist.index[0]
                price_2020 = hist['Close'].iloc[0]
            
            # Calculate key statistics from the displayed data
            volatility = hist['Close'].pct_change().std() * (252 ** 0.5) * 100  # Annualized
            avg_daily_volume = hist['Volume'].mean()
            max_price = hist['High'].max()
            min_price = hist['Low'].min()
            
            # Get latest values from displayed data
            latest_ma20 = hist['MA20'].iloc[-1] if 'MA20' in hist.columns and not pd.isna(hist['MA20'].iloc[-1]) else None
            latest_ma50 = hist['MA50'].iloc[-1] if 'MA50' in hist.columns and not pd.isna(hist['MA50'].iloc[-1]) else None
            latest_ma200 = hist['MA200'].iloc[-1] if 'MA200' in hist.columns and not pd.isna(hist['MA200'].iloc[-1]) else None
            
            # Count trend signals
            bullish_signals = sum([
                current_price > latest_ma20 if latest_ma20 else False,
                current_price > latest_ma50 if latest_ma50 else False,
                current_price > latest_ma200 if latest_ma200 else False,
                current_rsi < 70 and current_rsi > 50
            ])
            
            # Prepare comprehensive context
            price_2020_str = f"${price_2020:.2f}" if price_2020 else "N/A"
            change_from_2020 = ((current_price - price_2020) / price_2020 * 100) if price_2020 else None
            change_from_2020_str = f"+{change_from_2020:.1f}%" if change_from_2020 and change_from_2020 > 0 else f"{change_from_2020:.1f}%" if change_from_2020 else "N/A"
            
            # Pre-format values for context
            ma20_str = f"${latest_ma20:.2f}" if latest_ma20 else "N/A"
            ma50_str = f"${latest_ma50:.2f}" if latest_ma50 else "N/A"
            ma200_str = f"${latest_ma200:.2f}" if latest_ma200 else "N/A"
            volatility_str = f"{volatility:.1f}%"
            returns_1m_str = f"{returns_1m:+.2f}%" if returns_1m is not None else "N/A"
            returns_3m_str = f"{returns_3m:+.2f}%" if returns_3m is not None else "N/A"
            returns_1y_str = f"{returns_1y:+.2f}%" if returns_1y is not None else "N/A"
            
            # Build comprehensive prompt
            date_range_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            prompt = f"""You are a senior equity research analyst and technical analyst specializing in stock price analysis and technical indicators.

**{company_name} ({ticker}) - Price Analysis for Period: {date_range_str}**

**Long-Term Trend (2020-Present):**
- Price in 2020: {price_2020_str}
- Current Price: ${current_price:.2f}
- Total Change from 2020: {change_from_2020_str}

**Current Technical Indicators:**
- RSI (14): {current_rsi:.1f}
- 20-Day MA: {ma20_str}
- 50-Day MA: {ma50_str}
- 200-Day MA: {ma200_str}
- Bollinger Band Position: {bb_position*100:.1f}%
- Bullish Signals Active: {bullish_signals}/4

**Price Metrics:**
- Period High: ${max_price:.2f}
- Period Low: ${min_price:.2f}
- Annualized Volatility: {volatility_str}
- Average Daily Volume: {avg_daily_volume/1e6:.1f}M shares

**Performance Returns:**
- 1 Month: {returns_1m_str}
- 3 Month: {returns_3m_str}
- 1 Year: {returns_1y_str}

**Visual Context:**
The analysis includes four key visualizations:
1. **Candlestick Chart**: Shows price action with moving averages and Bollinger Bands
2. **RSI Chart**: Momentum indicator showing overbought/oversold conditions
3. **Volume Chart**: Trading volume patterns over time
4. **Returns Distribution**: Daily returns volatility pattern

**Your Task:**
Provide a comprehensive price and technical analysis (700-900 words) that covers:

1. **Long-Term Trend Analysis**: Analyze the {change_from_2020_str} change from 2020 baseline. What does this trajectory tell us over the {date_range_str} period? Reference the candlestick chart.

2. **Recent Price Action**: Examine the price trend over the selected period. Uptrend, downtrend, or consolidation? Reference candlestick chart.

3. **Technical Indicators**: Evaluate RSI at {current_rsi:.1f}, price vs moving averages, and Bollinger Band position. What do they signal?

4. **Volume Analysis**: Analyze trading volume patterns. Is volume confirming price moves?

5. **Support/Resistance**: Identify key levels near ${min_price:.2f} (support) and ${max_price:.2f} (resistance).

6. **Volatility Assessment**: With {volatility_str} volatility, assess risk level. Reference returns distribution.

7. **Performance Context**: Compare 1M, 3M, and 1Y returns. Which timeframe shows strength?

8. **Trading Signals**: Based on {bullish_signals}/4 signals and all indicators, what's the technical setup?

9. **Risk Considerations**: Key price levels to watch and potential volatility triggers.

10. **Investment Implications**: Guidance for traders vs long-term investors.

Be specific with prices and values, reference all four charts, and provide actionable insights."""

            try:
                # Generate AI insight using OpenAI
                import os
                from openai import OpenAI
                from dotenv import load_dotenv
                
                load_dotenv()
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a senior equity research analyst and technical analyst with deep expertise in chart analysis, technical indicators, and trading patterns. Provide comprehensive, data-driven insights that reference specific price levels and all visualizations."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                
                ai_insight = response.choices[0].message.content.strip()
                
                # Display AI insight in an attractive format
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 25px; border-radius: 10px; 
                            border-left: 5px solid #2196f3; line-height: 1.8; 
                            color: #212529; white-space: pre-wrap;">
                {ai_insight}
                </div>
                """, unsafe_allow_html=True)
                
                # Add disclaimer
                st.caption("💡 AI-generated price and technical analysis based on historical data and indicators. This analysis references all charts and metrics shown above. Technical analysis is one tool among many - conduct comprehensive research before investing. Past performance doesn't guarantee future results.")
                
            except Exception as e:
                st.warning(f"⚠️ Unable to generate AI price analysis: {str(e)}")
                st.info("Please ensure your OpenAI API key is properly configured in the .env file.")
    else:
        st.info("🔑 **AI Price Analysis Unavailable**: Configure your OpenAI API key in the .env file to enable comprehensive AI-powered price and technical analysis that examines all charts, indicators, and long-term trends from 2020 to present.")
