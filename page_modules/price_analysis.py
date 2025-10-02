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
from utils.text_utils import normalize_markdown_spacing


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
    st.plotly_chart(fig, width='stretch')
    
    # Technical indicators
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI Chart
        st.markdown("#### Relative Strength Index (RSI)")
        fig_rsi, hist = create_rsi_chart(hist)
        st.plotly_chart(fig_rsi, width='stretch')
        
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
        st.plotly_chart(fig_vol, width='stretch')
    
    with col2:
        st.subheader("Price Returns Distribution")
        returns = hist['Close'].pct_change().dropna()
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=returns, nbinsx=50, name='Daily Returns',
                                        marker_color='lightgreen'))
        fig_dist.update_layout(xaxis_title='Daily Return', yaxis_title='Frequency',
                              template='plotly_white', height=400)
        st.plotly_chart(fig_dist, width='stretch')
    
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
    st.markdown("### 🤖 AI-Powered Price Trend & News Analysis")
    st.markdown("Analysis of price movements and correlation with company events and market trends")
    
    # Import AI insights generator
    from utils.ai_insights_generator import AIInsightsGenerator
    from pathlib import Path
    
    # Initialize AI generator
    data_dir = Path("data")
    ai_generator = AIInsightsGenerator(data_dir)
    
    if ai_generator.enabled:
        with st.spinner("🧠 Analyzing price trends and company events..."):
            # Get company name from info
            company_name = info.get('longName', info.get('shortName', ticker))
            
            # Calculate key price metrics
            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            max_price = hist['High'].max()
            min_price = hist['Low'].min()
            total_return = ((end_price - start_price) / start_price) * 100
            
            # Find max price date and min price date
            max_price_date = hist['High'].idxmax().strftime('%Y-%m-%d')
            min_price_date = hist['Low'].idxmin().strftime('%Y-%m-%d')
            max_price_actual = hist.loc[hist['High'].idxmax(), 'Close']
            min_price_actual = hist.loc[hist['Low'].idxmin(), 'Close']
            
            # Identify significant price movements (>5% in a day)
            hist['Daily_Change'] = hist['Close'].pct_change() * 100
            big_moves = hist[abs(hist['Daily_Change']) > 5].copy()
            big_moves_list = []
            for date, row in big_moves.head(15).iterrows():  # Top 15 significant moves
                big_moves_list.append(f"  • {date.strftime('%Y-%m-%d')}: {row['Daily_Change']:+.2f}% (Close: ${row['Close']:.2f})")
            
            # Get yearly performance if data spans multiple years
            yearly_performance = []
            years_in_data = sorted(hist.index.year.unique())
            for year in years_in_data:
                year_data = hist[hist.index.year == year]
                if len(year_data) > 0:
                    year_start_price = year_data['Close'].iloc[0]
                    year_end_price = year_data['Close'].iloc[-1]
                    year_high = year_data['High'].max()
                    year_low = year_data['Low'].min()
                    year_return = ((year_end_price - year_start_price) / year_start_price) * 100
                    yearly_performance.append(
                        f"  • {year}: Started at ${year_start_price:.2f}, ended at ${year_end_price:.2f} "
                        f"(Return: {year_return:+.2f}%, High: ${year_high:.2f}, Low: ${year_low:.2f})"
                    )
            
            # Find local peaks and troughs (significant turning points)
            # Look for points where price changed direction significantly
            hist['Price_Change_30d'] = hist['Close'].pct_change(30) * 100
            turning_points = []
            
            # Find peaks (local maxima)
            for i in range(30, len(hist)-30):
                window = hist.iloc[i-30:i+30]
                if hist['Close'].iloc[i] == window['Close'].max():
                    date = hist.index[i].strftime('%Y-%m-%d')
                    price = hist['Close'].iloc[i]
                    turning_points.append(f"  • Peak on {date}: ${price:.2f}")
            
            # Find troughs (local minima)
            for i in range(30, len(hist)-30):
                window = hist.iloc[i-30:i+30]
                if hist['Close'].iloc[i] == window['Close'].min():
                    date = hist.index[i].strftime('%Y-%m-%d')
                    price = hist['Close'].iloc[i]
                    turning_points.append(f"  • Trough on {date}: ${price:.2f}")
            
            # Limit to top 10 most significant
            turning_points = turning_points[:10]
            
            # Build comprehensive prompt focused on trends and events
            date_range_str = f"{start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}"
            prompt = f"""You are a senior equity research analyst specializing in analyzing stock price trends and correlating them with company news and market events.

**{company_name} ({ticker}) - Price Trend Analysis**

**CRITICAL: Your analysis must be specifically for the period from {date_range_str}. State this timeframe at the beginning.**

**⚠️ STRICT DATA ACCURACY REQUIREMENTS:**
- ONLY reference dates and prices explicitly provided in the data below
- DO NOT make up or estimate specific dates or price levels
- If discussing general trends, use approximate timeframes (e.g., "mid-2022" not "November 11, 2022")
- When referencing prices, ONLY use the exact figures provided below
- If you don't have specific data about an event, acknowledge the limitation

**Price Performance Summary:**
- Starting Price ({start_date.strftime('%Y-%m-%d')}): ${start_price:.2f}
- Ending Price ({end_date.strftime('%Y-%m-%d')}): ${end_price:.2f}
- Total Return: {total_return:+.2f}%
- Peak Price: ${max_price:.2f} on {max_price_date} (Close: ${max_price_actual:.2f})
- Lowest Price: ${min_price:.2f} on {min_price_date} (Close: ${min_price_actual:.2f})

**Yearly Performance Breakdown:**
{chr(10).join(yearly_performance) if yearly_performance else "  Single year analysis"}

**Major Turning Points (Local Peaks and Troughs):**
{chr(10).join(turning_points) if turning_points else "  No significant local extrema detected"}

**Significant Single-Day Movements (>5%):**
{chr(10).join(big_moves_list) if big_moves_list else "  No major single-day movements >5%"}

**Your Task:**
Provide a comprehensive, narrative-style price trend and event analysis (600-800 words) in a flowing, paragraph format. ONLY use the data provided above.

Write your analysis as a cohesive narrative story that:

1. **Opens with Context**: Begin by stating "This analysis covers the period from {date_range_str}" and immediately introduce the overall price journey - starting at ${start_price:.2f}, ending at ${end_price:.2f}, for a total return of {total_return:+.2f}%.

2. **Tells the Price Story Chronologically**: Walk through the price movements year by year or phase by phase, weaving together:
   - The yearly performance figures provided (actual start/end prices and returns for each year)
   - The key turning points (peaks and troughs with their exact dates and prices)
   - The significant daily moves (>5% movements with exact dates and percentage changes)
   - Connect these data points into a flowing narrative about what happened and why it likely occurred

3. **Provides Context Without Inventing Details**: 
   - Discuss likely catalysts in GENERAL terms (e.g., "likely related to earnings season," "possibly driven by sector-wide trends," "coinciding with broader market volatility")
   - Reference broader market and industry context for each period
   - Use approximate timeframes (e.g., "early 2022," "throughout 2023," "the second half of the year") when specific dates aren't provided
   - DO NOT invent specific event dates, news items, or prices not in the data

4. **Highlights Key Observations**: Throughout the narrative, naturally incorporate:
   - The price range: from the low of ${min_price:.2f} on {min_price_date} to the peak of ${max_price:.2f} on {max_price_date}
   - The {((max_price - min_price) / min_price * 100):.1f}% range during the period
   - Where the current price (${end_price:.2f}) sits relative to these extremes
   - Patterns that emerge from the data (volatility, trends, reversals)

5. **Concludes with Synthesis**: End with a summary paragraph that ties together the overall trajectory, the {total_return:+.2f}% return, and what patterns or themes emerged during this period.

**Formatting Requirements:**
- Write in flowing paragraphs (3-6 paragraphs total), NOT bullet points or numbered sections
- DO NOT use multiple heading levels (###) within your response - write as continuous prose
- Bold key figures (dates, prices, percentages) to make them stand out: **$150.25**, **+25.3%**, **March 15, 2023**
- Use smooth transitions between paragraphs to maintain narrative flow
- Write in an engaging, professional tone as if briefing an investor

**CRITICAL RULES - AVOID HALLUCINATION:**
- ✅ DO: Use exact dates/prices from the data above
- ✅ DO: Discuss general industry trends and macro events
- ✅ DO: Use approximate timeframes like "early 2022" or "throughout 2023"
- ❌ DO NOT: Invent specific dates like "November 11, 2022" unless it appears in the data
- ❌ DO NOT: Make up specific price levels not provided above
- ❌ DO NOT: Claim specific news events without qualification (use "likely" or "possibly")
- ❌ DO NOT: Discuss technical indicators (RSI, moving averages, Bollinger Bands)
- ❌ DO NOT: Repeat metrics already shown on the page above"""

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
                            "content": "You are a senior equity research analyst writing narrative investment reports. Your TOP PRIORITY is DATA ACCURACY - only reference dates and prices explicitly provided in the prompt. Write in flowing, paragraph-based narrative format, NOT as bullet points or multiple sections with headings. When discussing events, use general terms and timeframes unless you have specific data. Never make up or estimate specific dates or prices. If you lack specific information, acknowledge it. Tell the story of the stock's price movement as a cohesive narrative with smooth transitions between ideas."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                
                ai_insight = normalize_markdown_spacing(response.choices[0].message.content.strip())
                
                # Display AI insight using Streamlit's native styling
                # Add custom CSS for the container but render markdown separately
                st.markdown("""
                <style>
                div[data-testid="stMarkdownContainer"] > div.ai-analysis-container {
                    background-color: #e3f2fd;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid #2196f3;
                    margin: 10px 0;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Use st.info for automatic styling, or plain markdown
                st.markdown("---")
                st.info("📊 AI-Generated Price Trend Analysis")
                st.markdown(ai_insight)
                st.markdown("---")
                
                # Add disclaimer
                st.caption("💡 AI-generated price trend analysis based on historical price movements and correlation with company events and market conditions. This narrative explains what happened during the selected timeframe and potential catalysts. Always verify events with official sources and conduct comprehensive research before investing. Past performance doesn't guarantee future results.")
                
            except Exception as e:
                st.warning(f"⚠️ Unable to generate AI price analysis: {str(e)}")
                st.info("Please ensure your OpenAI API key is properly configured in the .env file.")
    else:
        st.info("🔑 **AI Price Trend Analysis Unavailable**: Configure your OpenAI API key in the .env file to enable AI-powered analysis of price trends and correlation with company events, news, and market catalysts during your selected timeframe.")
