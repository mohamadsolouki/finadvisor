import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path
import yfinance as yf
import numpy as np
import time

# Add utils directory to path
sys.path.append(str(Path(__file__).parent))

from utils.data_analyzer import FinancialAnalyzer
from utils.report_generator import ReportGenerator

# Ticker symbol for Qualcomm
TICKER = "QCOM"

# Page configuration
st.set_page_config(
    page_title="Qualcomm Financial Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_categorize_data(file_path):
    """Load and categorize financial data into sections"""
    df = pd.read_csv(file_path)
    
    categories = {
        'Income Statement': [],
        'Balance Sheet': [],
        'Cash Flow': [],
        'Growth Ratios': [],
        'Equity Ratios': [],
        'Profitability Ratios': [],
        'Cost Ratios': [],
        'Liquidity Ratios': [],
        'Leverage Ratios': [],
        'Efficiency Ratios': []
    }
    
    current_category = None
    
    # Map CSV headers to category names
    header_to_category = {
        'Income Statements': 'Income Statement',
        'Balance Sheet': 'Balance Sheet',
        'Cash Flow': 'Cash Flow',
        'Key Ratios': None  # This is followed by subcategories
    }
    
    for idx, row in df.iterrows():
        param = str(row['Parameters']).strip()
        
        # Check if this is a main category header (Income Statements, Balance Sheet, etc.)
        if param in header_to_category:
            if header_to_category[param]:
                current_category = header_to_category[param]
            # For Key Ratios, wait for subcategory
            continue
        elif param in categories.keys():
            # This handles subcategories under Key Ratios
            current_category = param
        elif current_category and param != 'nan' and not pd.isna(row['Parameters']):
            categories[current_category].append(idx)
    
    # Create separate dataframes for each category
    categorized_data = {}
    for category, indices in categories.items():
        if indices:
            categorized_data[category] = df.loc[indices].reset_index(drop=True)
    
    return df, categorized_data


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


def display_category_data(category_name, data, analyzer):
    """Display data for a specific category with visualizations"""
    st.markdown(f'<div class="section-header">{category_name}</div>', unsafe_allow_html=True)
    
    # Display the data table
    st.dataframe(data, use_container_width=True, height=min(len(data) * 35 + 38, 400))
    
    # Create visualizations based on category
    col1, col2 = st.columns(2)
    
    with col1:
        if len(data) > 0:
            # Single metric trend
            metric_row = data.iloc[0:1]
            fig = create_trend_chart(metric_row, f"{data.iloc[0]['Parameters']} Trend", data.iloc[0]['Parameters'])
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if len(data) > 1:
            # Multiple metrics comparison
            comparison_data = data.iloc[:min(5, len(data))]
            fig = create_comparison_chart(comparison_data, f"{category_name} - Top Metrics Comparison")
            st.plotly_chart(fig, use_container_width=True)
    
    # Display insights
    insights = analyzer.get_category_insights(category_name, data)
    if insights:
        st.info(f"**Key Insights:** {insights}")
    
    st.markdown("---")


@st.cache_data(ttl=7200)  # Cache for 2 hours to reduce API calls
def get_stock_data(ticker, period="2y"):
    """Fetch stock data from Yahoo Finance with caching"""
    try:
        time.sleep(0.2)  # Small delay to avoid rapid consecutive calls
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        return stock, hist, info
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")
        st.info("💡 Tip: Click 'Clear Cache & Refresh Data' in the sidebar and wait a few minutes before trying again.")
        return None, None, None


@st.cache_data(ttl=7200)  # Cache for 2 hours
def get_stock_info(ticker):
    """Fetch only stock info to reduce API calls"""
    try:
        time.sleep(0.2)  # Small delay to avoid rapid consecutive calls
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        st.warning(f"⚠️ Unable to fetch current market data. Using cached data if available.")
        if "429" in str(e) or "Too Many Requests" in str(e):
            st.error("🚫 **Rate Limit Reached**: Yahoo Finance is temporarily blocking requests. Please wait 5-10 minutes.")
        return {}


@st.cache_data(ttl=7200)  # Cache for 2 hours
def get_historical_data(ticker, period="2y"):
    """Fetch only historical price data"""
    try:
        time.sleep(0.2)  # Small delay to avoid rapid consecutive calls
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        st.warning(f"⚠️ Unable to fetch historical price data.")
        if "429" in str(e) or "Too Many Requests" in str(e):
            st.error("🚫 **Rate Limit Reached**: Please wait 5-10 minutes before refreshing.")
        return pd.DataFrame()


@st.cache_data(ttl=7200)  # Cache for 2 hours
def get_esg_data(ticker):
    """Fetch ESG data separately with caching"""
    try:
        time.sleep(0.2)  # Small delay to avoid rapid consecutive calls
        stock = yf.Ticker(ticker)
        esg_data = stock.sustainability
        return esg_data
    except Exception as e:
        return None


@st.cache_data(ttl=7200)  # Cache for 2 hours
def get_competitor_data(ticker_list):
    """Fetch competitor data in batch with caching and rate limiting"""
    comparison_data = []
    
    for idx, (ticker_sym, name) in enumerate(ticker_list.items()):
        try:
            # Add delay between requests to avoid rate limiting (except for first request)
            if idx > 0:
                time.sleep(0.5)  # 500ms delay between requests
            
            comp_stock = yf.Ticker(ticker_sym)
            comp_info = comp_stock.info
            comparison_data.append({
                'Company': name,
                'Ticker': ticker_sym,
                'Market Cap (B)': comp_info.get('marketCap', 0) / 1e9,
                'P/E Ratio': comp_info.get('trailingPE', 0),
                'P/B Ratio': comp_info.get('priceToBook', 0),
                'Profit Margin (%)': comp_info.get('profitMargins', 0) * 100,
                'Operating Margin (%)': comp_info.get('operatingMargins', 0) * 100,
                'ROE (%)': comp_info.get('returnOnEquity', 0) * 100,
                'Debt/Equity': comp_info.get('debtToEquity', 0),
                'Dividend Yield (%)': comp_info.get('dividendYield', 0) * 100 if comp_info.get('dividendYield') else 0
            })
        except Exception as e:
            # Skip failed tickers silently (they're already cached if they failed once)
            continue
    
    return comparison_data


def display_price_analysis(ticker, cached_info=None):
    """Display price analysis page"""
    st.subheader("📈 Stock Price Analysis & Technical Indicators")
    
    # Time period selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### Market Data")
    with col2:
        period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
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
    
    # Calculate moving averages
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()
    
    # Calculate Bollinger Bands
    hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
    hist['BB_Std'] = hist['Close'].rolling(window=20).std()
    hist['BB_Upper'] = hist['BB_Middle'] + (hist['BB_Std'] * 2)
    hist['BB_Lower'] = hist['BB_Middle'] - (hist['BB_Std'] * 2)
    
    # Calculate RSI
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))
    
    # Main price chart with candlestick
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
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], name='20-Day MA',
                             line=dict(color='#ffa726', width=1.5)))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], name='50-Day MA',
                             line=dict(color='#42a5f5', width=1.5)))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], name='200-Day MA',
                             line=dict(color='#ef5350', width=2)))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Upper'], name='BB Upper',
                             line=dict(color='gray', width=1, dash='dot'),
                             showlegend=False, opacity=0.5))
    fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Lower'], name='BB Lower',
                             line=dict(color='gray', width=1, dash='dot'),
                             fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                             showlegend=False, opacity=0.5))
    
    fig.update_layout(title=f'{ticker} Stock Price with Technical Indicators',
                      xaxis_title='Date', yaxis_title='Price (USD)',
                      hovermode='x unified', template='plotly_white', height=600,
                      xaxis_rangeslider_visible=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI Chart
        st.markdown("#### Relative Strength Index (RSI)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], 
                                      name='RSI', line=dict(color='purple', width=2)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                          annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", 
                          annotation_text="Oversold (30)")
        fig_rsi.update_layout(yaxis_title='RSI', xaxis_title='Date',
                             template='plotly_white', height=300)
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


def display_esg_analysis(ticker, cached_info=None):
    """Display ESG analysis page"""
    st.subheader("🌱 ESG (Environmental, Social, Governance) Analysis")
    
    # Use cached info if available
    info = cached_info if cached_info else get_stock_info(ticker)
    
    if not info:
        st.warning("Unable to fetch company data. Please try again later.")
        st.info("Yahoo Finance may be rate-limiting requests. Data is cached for 2 hours once successfully loaded.")
        return
    
    try:
        # Try to get ESG scores using cached function
        esg_data = get_esg_data(ticker)
        
        if esg_data is not None and not esg_data.empty:
            st.success(f"✅ ESG data available for {ticker}")
            
            # Display ESG scores
            col1, col2, col3, col4 = st.columns(4)
            
            esg_score = esg_data.loc['totalEsg'].iloc[0] if 'totalEsg' in esg_data.index else None
            env_score = esg_data.loc['environmentScore'].iloc[0] if 'environmentScore' in esg_data.index else None
            social_score = esg_data.loc['socialScore'].iloc[0] if 'socialScore' in esg_data.index else None
            gov_score = esg_data.loc['governanceScore'].iloc[0] if 'governanceScore' in esg_data.index else None
            
            with col1:
                if esg_score is not None:
                    st.metric("Total ESG Score", f"{esg_score:.1f}")
            
            with col2:
                if env_score is not None:
                    st.metric("Environment Score", f"{env_score:.1f}")
            
            with col3:
                if social_score is not None:
                    st.metric("Social Score", f"{social_score:.1f}")
            
            with col4:
                if gov_score is not None:
                    st.metric("Governance Score", f"{gov_score:.1f}")
            
            st.markdown("---")
            
            # ESG score visualization
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if all(score is not None for score in [env_score, social_score, gov_score]):
                    fig = go.Figure(data=[
                        go.Bar(name='ESG Components', 
                               x=['Environment', 'Social', 'Governance'],
                               y=[env_score, social_score, gov_score],
                               marker_color=['#28a745', '#007bff', '#fd7e14'],
                               text=[f'{env_score:.1f}', f'{social_score:.1f}', f'{gov_score:.1f}'],
                               textposition='outside')
                    ])
                    fig.update_layout(title='ESG Score Breakdown', yaxis_title='Score',
                                     template='plotly_white', height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # ESG interpretation
                st.markdown("### Score Guide")
                st.info("""
                **ESG Scores (0-100):**
                - **0-20**: Poor
                - **20-40**: Below Avg
                - **40-60**: Average
                - **60-80**: Good
                - **80-100**: Excellent
                
                *Note: Lower scores may indicate better performance in some rating systems*
                """)
            
            st.markdown("---")
            
            # Display full ESG dataframe
            st.subheader("📋 Detailed ESG Metrics")
            st.dataframe(esg_data, use_container_width=True)
            
            # Additional context from company info
            st.markdown("---")
            st.subheader("🏢 Company Context")
            col1, col2 = st.columns(2)
            
            with col1:
                sector = info.get('sector', 'N/A')
                industry = info.get('industry', 'N/A')
                employees = info.get('fullTimeEmployees', 'N/A')
                
                st.write(f"**Sector:** {sector}")
                st.write(f"**Industry:** {industry}")
                st.write(f"**Full-time Employees:** {employees:,}" if isinstance(employees, int) else f"**Full-time Employees:** {employees}")
            
            with col2:
                # ESG controversies if available
                if 'controversyLevel' in esg_data.index:
                    controversy = esg_data.loc['controversyLevel'].iloc[0]
                    st.write(f"**Controversy Level:** {controversy}")
                
                # Additional metrics
                if 'highestControversy' in esg_data.index:
                    highest_controversy = esg_data.loc['highestControversy'].iloc[0]
                    st.write(f"**Highest Controversy:** {highest_controversy}")
            
        else:
            st.warning(f"⚠️ ESG data is not currently available for {ticker} through Yahoo Finance.")
            st.info("""
            **Why ESG data might be unavailable:**
            - Company may not be large enough to be rated
            - ESG ratings may not be publicly disclosed
            - Data provider limitations
            
            **Alternative ESG data sources:**
            - MSCI ESG Ratings
            - Sustainalytics
            - Bloomberg ESG Data
            - CDP (Carbon Disclosure Project)
            """)
            
    except Exception as e:
        st.warning(f"⚠️ ESG data is not currently available for {ticker}.")
        st.info(f"ESG data availability varies by company and data provider.")
        with st.expander("Technical Details"):
            st.error(f"Error: {str(e)}")


def display_industry_benchmarking(ticker, cached_info=None):
    """Display industry benchmarking page"""
    st.subheader("📊 Industry Benchmarking & Peer Comparison")
    
    # Use cached info if available
    info = cached_info if cached_info else get_stock_info(ticker)
    
    if not info:
        st.warning("Unable to fetch company data. Please try again later.")
        st.info("Yahoo Finance may be rate-limiting requests. Data is cached for 2 hours once successfully loaded.")
        return
    
    # Company info header
    col1, col2, col3 = st.columns(3)
    sector = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')
    
    with col1:
        st.metric("🏢 Sector", sector)
    with col2:
        st.metric("🏭 Industry", industry)
    with col3:
        employees = info.get('fullTimeEmployees', 'N/A')
        st.metric("👥 Employees", f"{employees:,}" if isinstance(employees, int) else employees)
    
    st.markdown("---")
    
    # Key metrics comparison
    st.subheader("📈 Company Metrics Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        market_cap = info.get('marketCap', 0) / 1e9
        st.metric("Market Cap", f"${market_cap:.2f}B")
    
    with col2:
        pe_ratio = info.get('trailingPE', 0)
        st.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
    
    with col3:
        pb_ratio = info.get('priceToBook', 0)
        st.metric("P/B Ratio", f"{pb_ratio:.2f}" if pb_ratio else "N/A")
    
    with col4:
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        st.metric("Dividend Yield", f"{dividend_yield:.2f}%")
    
    st.markdown("---")
    
    # Profitability metrics
    st.subheader("💰 Profitability & Efficiency Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        profit_margin = info.get('profitMargins', 0) * 100
        st.metric("Profit Margin", f"{profit_margin:.2f}%")
    
    with col2:
        operating_margin = info.get('operatingMargins', 0) * 100
        st.metric("Operating Margin", f"{operating_margin:.2f}%")
    
    with col3:
        roe = info.get('returnOnEquity', 0) * 100
        st.metric("Return on Equity", f"{roe:.2f}%")
    
    with col4:
        roa = info.get('returnOnAssets', 0) * 100
        st.metric("Return on Assets", f"{roa:.2f}%")
    
    # Additional financial metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        revenue = info.get('totalRevenue', 0) / 1e9
        st.metric("Total Revenue", f"${revenue:.2f}B")
    
    with col2:
        revenue_growth = info.get('revenueGrowth', 0) * 100
        st.metric("Revenue Growth", f"{revenue_growth:.2f}%")
    
    with col3:
        earnings_growth = info.get('earningsGrowth', 0) * 100
        st.metric("Earnings Growth", f"{earnings_growth:.2f}%")
    
    with col4:
        debt_to_equity = info.get('debtToEquity', 0)
        st.metric("Debt/Equity", f"{debt_to_equity:.2f}")
    
    st.markdown("---")
    
    # Competitor comparison
    st.subheader("🔍 Peer Comparison Analysis")
    st.info("Comparing with major semiconductor companies")
    
    # List of competitors
    competitors = {
        'QCOM': 'Qualcomm',
        'NVDA': 'NVIDIA',
        'AVGO': 'Broadcom',
        'TXN': 'Texas Instruments',
        'AMD': 'AMD',
        'INTC': 'Intel'
    }
    
    # Use cached competitor data
    with st.spinner("Loading peer company data..."):
        comparison_data = get_competitor_data(competitors)
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        
        # Highlight the target company
        def highlight_target(row):
            if row['Ticker'] == ticker:
                return ['background-color: #ffffcc'] * len(row)
            return [''] * len(row)
        
        st.dataframe(comp_df.style.apply(highlight_target, axis=1).format({
            'Market Cap (B)': '${:.2f}B',
            'P/E Ratio': '{:.2f}',
            'P/B Ratio': '{:.2f}',
            'Profit Margin (%)': '{:.2f}%',
            'Operating Margin (%)': '{:.2f}%',
            'ROE (%)': '{:.2f}%',
            'Debt/Equity': '{:.2f}',
            'Dividend Yield (%)': '{:.2f}%'
        }), use_container_width=True)
        
        st.markdown("---")
        
        # Visualization
        st.subheader("📊 Visual Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(comp_df, x='Company', y='Market Cap (B)',
                        title='Market Cap Comparison',
                        color='Market Cap (B)', color_continuous_scale='blues',
                        text='Market Cap (B)')
            fig.update_traces(texttemplate='$%{text:.1f}B', textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(comp_df, x='Company', y='Profit Margin (%)',
                        title='Profit Margin Comparison',
                        color='Profit Margin (%)', color_continuous_scale='greens',
                        text='Profit Margin (%)')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(comp_df, x='Company', y='ROE (%)',
                        title='Return on Equity Comparison',
                        color='ROE (%)', color_continuous_scale='oranges',
                        text='ROE (%)')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(comp_df, x='Company', y='P/E Ratio',
                        title='P/E Ratio Comparison',
                        color='P/E Ratio', color_continuous_scale='purples',
                        text='P/E Ratio')
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("Unable to fetch peer comparison data.")


def display_risk_analysis(ticker, cached_info=None):
    """Display risk analysis page"""
    st.subheader("⚠️ Risk Analysis & Portfolio Metrics")
    
    # Use cached data
    hist = get_historical_data(ticker, period="5y")
    info = cached_info if cached_info else get_stock_info(ticker)
    
    if hist is None or hist.empty:
        st.warning("Unable to fetch historical data for risk analysis. Please try again later.")
        st.info("Yahoo Finance may be rate-limiting requests. Data is cached for 2 hours once successfully loaded.")
        return
    
    # Calculate returns
    returns = hist['Close'].pct_change().dropna()
    
    # Risk metrics header
    st.markdown("### 📊 Key Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        beta = info.get('beta', 0)
        st.metric("Beta", f"{beta:.2f}" if beta else "N/A")
    
    with col2:
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        st.metric("Volatility (Annual)", f"{volatility:.2f}%")
    
    with col3:
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    with col4:
        max_drawdown = ((hist['Close'] / hist['Close'].cummax()) - 1).min() * 100
        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    
    # Additional risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Downside deviation
        negative_returns = returns[returns < 0]
        downside_dev = negative_returns.std() * np.sqrt(252) * 100
        st.metric("Downside Deviation", f"{downside_dev:.2f}%")
    
    with col2:
        # Sortino ratio
        sortino = (returns.mean() * 252) / (negative_returns.std() * np.sqrt(252)) if len(negative_returns) > 0 else 0
        st.metric("Sortino Ratio", f"{sortino:.2f}")
    
    with col3:
        # Positive days percentage
        positive_days = (returns > 0).sum() / len(returns) * 100
        st.metric("Positive Days", f"{positive_days:.1f}%")
    
    with col4:
        # Average gain/loss ratio
        avg_gain = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
        gain_loss_ratio = avg_gain / avg_loss if avg_loss != 0 else 0
        st.metric("Gain/Loss Ratio", f"{gain_loss_ratio:.2f}")
    
    st.markdown("---")
    
    # Value at Risk (VaR) and Expected Shortfall
    st.markdown("### 📉 Value at Risk (VaR) & Expected Shortfall (CVaR)")
    
    confidence_levels = [0.90, 0.95, 0.99]
    var_data = []
    
    for conf in confidence_levels:
        var_pct = np.percentile(returns, (1 - conf) * 100)
        # Expected Shortfall (CVaR) - average of returns worse than VaR
        cvar_pct = returns[returns <= var_pct].mean()
        
        var_data.append({
            'Confidence Level': f"{conf*100}%",
            'Daily VaR (%)': var_pct * 100,
            'Daily CVaR (%)': cvar_pct * 100,
            'Annual VaR (%)': var_pct * np.sqrt(252) * 100,
            'Annual CVaR (%)': cvar_pct * np.sqrt(252) * 100
        })
    
    var_df = pd.DataFrame(var_data)
    st.dataframe(var_df.round(3), use_container_width=True)
    
    st.info("""
    **VaR (Value at Risk):** Maximum expected loss over a given time period at a certain confidence level.
    
    **CVaR (Conditional VaR/Expected Shortfall):** Average loss in the worst-case scenarios beyond VaR threshold.
    """)
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Rolling volatility
        st.markdown("### 📈 Rolling 30-Day Volatility")
        
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol,
                                 name='Rolling Volatility',
                                 line=dict(color='#e74c3c', width=2),
                                 fill='tozeroy'))
        fig.add_hline(y=volatility, line_dash="dash", line_color="gray", 
                      annotation_text=f"Avg: {volatility:.2f}%")
        fig.update_layout(xaxis_title='Date', yaxis_title='Volatility (%)',
                         template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Returns distribution
        st.markdown("### 📊 Returns Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns * 100, nbinsx=50, 
                                    name='Daily Returns',
                                    marker_color='#3498db'))
        
        # Add VaR lines
        var_95 = np.percentile(returns, 5) * 100
        fig.add_vline(x=var_95, line_dash="dash", line_color="red",
                      annotation_text="VaR 95%", annotation_position="top")
        
        fig.update_layout(xaxis_title='Daily Return (%)', yaxis_title='Frequency',
                         template='plotly_white', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Drawdown analysis
    st.markdown("### 📉 Drawdown Analysis")
    
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max - 1) * 100
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown,
                                 fill='tozeroy', name='Drawdown',
                                 line=dict(color='#c0392b', width=2),
                                 fillcolor='rgba(192, 57, 43, 0.3)'))
        fig.update_layout(xaxis_title='Date', yaxis_title='Drawdown (%)',
                         template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Drawdown Statistics")
        
        # Calculate drawdown statistics
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdown.items():
            if dd < -1 and not in_drawdown:  # Starting a drawdown
                in_drawdown = True
                start_date = date
            elif dd >= -1 and in_drawdown:  # Ending a drawdown
                in_drawdown = False
                if start_date:
                    drawdown_periods.append((date - start_date).days)
        
        avg_recovery_days = np.mean(drawdown_periods) if drawdown_periods else 0
        max_recovery_days = max(drawdown_periods) if drawdown_periods else 0
        
        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        st.metric("Avg Recovery (days)", f"{int(avg_recovery_days)}")
        st.metric("Max Recovery (days)", f"{int(max_recovery_days)}")
        st.metric("# of Drawdowns", f"{len(drawdown_periods)}")
    
    st.markdown("---")
    
    # Risk interpretation
    st.markdown("### 📚 Risk Metrics Interpretation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Beta: {beta:.2f}**
        - **< 1:** Less volatile than market
        - **= 1:** Moves with market
        - **> 1:** More volatile than market
        
        *Current: {'Low volatility' if beta < 1 else 'High volatility' if beta > 1 else 'Market volatility'}*
        """)
    
    with col2:
        st.info(f"""
        **Sharpe Ratio: {sharpe:.2f}**
        - **< 1:** Poor risk-adjusted return
        - **1-2:** Good
        - **> 2:** Excellent
        
        *Current: {'Excellent' if sharpe > 2 else 'Good' if sharpe > 1 else 'Poor'}*
        """)
    
    with col3:
        st.info(f"""
        **Sortino Ratio: {sortino:.2f}**
        - Similar to Sharpe but focuses on downside risk
        - **> 2:** Excellent
        - **1-2:** Good
        - **< 1:** Poor
        
        *Current: {'Excellent' if sortino > 2 else 'Good' if sortino > 1 else 'Poor'}*
        """)



def main():
    # Header
    st.markdown('<div class="main-header">📊 Qualcomm Financial Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Initialize session state for data persistence
    if 'last_fetch_time' not in st.session_state:
        st.session_state.last_fetch_time = None
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    # Load data
    data_path = Path(__file__).parent / "data" / "QCOMfinancials.csv"
    
    try:
        full_data, categorized_data = load_and_categorize_data(data_path)
        analyzer = FinancialAnalyzer(categorized_data)
        report_gen = ReportGenerator(full_data, categorized_data, analyzer)
        
        # Sidebar options - Updated navigation structure
        page = st.sidebar.radio(
            "Select View",
            ["Executive Summary", "Financial Analysis", "Price Analysis", 
             "ESG Analysis", "Industry Benchmarking", "Risk Analysis", "Custom Analysis"]
        )
        
        st.sidebar.markdown("---")
        
        # Data refresh section
        st.sidebar.subheader("⚙️ Data Management")
        
        if st.sidebar.button("🔄 Clear Cache & Refresh Data"):
            st.cache_data.clear()
            st.session_state.last_fetch_time = None
            st.sidebar.success("✅ Cache cleared!")
            st.sidebar.warning("⏱️ Please wait 30 seconds before navigating to avoid rate limits.")
            time.sleep(1)
            st.rerun()
        
        # Show last fetch time
        if st.session_state.last_fetch_time:
            time_diff = datetime.now() - st.session_state.last_fetch_time
            minutes = time_diff.seconds // 60
            st.sidebar.caption(f"📅 Last refreshed: {minutes} min ago" if minutes > 0 else "📅 Just refreshed")
        
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **📊 Data Sources:**
        - Financial Statements: CSV file
        - Market Data: Yahoo Finance API
        
        **⚡ Performance:**
        - Data cached for 2 hours
        - Reduces API calls
        - Avoids rate limiting
        
        **💡 Tips:**
        - If you see errors, wait 5-10 min
        - Use refresh button sparingly
        - Data updates automatically
        """)
        
        # Fetch Yahoo Finance data once and cache it
        if page in ["Price Analysis", "ESG Analysis", "Industry Benchmarking", "Risk Analysis"]:
            with st.spinner("Loading market data..."):
                cached_info = get_stock_info(TICKER)
                if not st.session_state.last_fetch_time:
                    st.session_state.last_fetch_time = datetime.now()
        else:
            cached_info = None
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Export Options")
        
        # Export buttons
        if st.sidebar.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                pdf_buffer = report_gen.generate_pdf_report()
                st.sidebar.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_buffer,
                    file_name=f"Qualcomm_Financial_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        if st.sidebar.button("📊 Generate Excel Report", use_container_width=True):
            with st.spinner("Generating Excel report..."):
                excel_buffer = report_gen.generate_excel_report()
                st.sidebar.download_button(
                    label="⬇️ Download Excel",
                    data=excel_buffer,
                    file_name=f"Qualcomm_Financial_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        # Main content based on page selection
        if page == "Executive Summary":
            st.subheader("Executive Summary")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Get latest year data - extract years from any available data
            latest_year = None
            prev_year = None
            
            # Try to get years from income data first, then fallback to other categories
            income_data = categorized_data.get('Income Statement')
            if income_data is not None and len(income_data) > 0:
                years = [col for col in income_data.columns if col not in ['Parameters', 'Currency']]
                if len(years) > 0:
                    latest_year = years[-1]
                    prev_year = years[-2] if len(years) > 1 else latest_year
            
            # If years not found in income data, try equity or profitability data
            if latest_year is None:
                for category in ['Equity Ratios', 'Profitability Ratios', 'Liquidity Ratios']:
                    data = categorized_data.get(category)
                    if data is not None and len(data) > 0:
                        years = [col for col in data.columns if col not in ['Parameters', 'Currency']]
                        if len(years) > 0:
                            latest_year = years[-1]
                            prev_year = years[-2] if len(years) > 1 else latest_year
                            break
            
            # Revenue metrics
            if income_data is not None and len(income_data) > 0 and latest_year is not None:
                # Revenue
                revenue_row = income_data[income_data['Parameters'] == 'Total Revenue']
                if not revenue_row.empty:
                    current_revenue = float(str(revenue_row[latest_year].iloc[0]).replace(',', ''))
                    prev_revenue = float(str(revenue_row[prev_year].iloc[0]).replace(',', ''))
                    revenue_change = ((current_revenue - prev_revenue) / prev_revenue) * 100
                    
                    with col1:
                        st.metric("Total Revenue (USD M)", f"${current_revenue:,.0f}", 
                                 f"{revenue_change:+.2f}%")
                
                # Net Income
                net_income_row = income_data[income_data['Parameters'] == 'Net Income']
                if not net_income_row.empty:
                    current_ni = float(str(net_income_row[latest_year].iloc[0]).replace(',', ''))
                    prev_ni = float(str(net_income_row[prev_year].iloc[0]).replace(',', ''))
                    ni_change = ((current_ni - prev_ni) / prev_ni) * 100
                    
                    with col2:
                        st.metric("Net Income (USD M)", f"${current_ni:,.0f}", 
                                 f"{ni_change:+.2f}%")
            
            # EPS
            equity_data = categorized_data.get('Equity Ratios')
            if equity_data is not None and len(equity_data) > 0 and latest_year is not None:
                eps_row = equity_data[equity_data['Parameters'] == 'EPS (Earnings per Share)']
                if not eps_row.empty:
                    current_eps = float(str(eps_row[latest_year].iloc[0]).replace(',', ''))
                    prev_eps = float(str(eps_row[prev_year].iloc[0]).replace(',', ''))
                    eps_change = ((current_eps - prev_eps) / prev_eps) * 100
                    
                    with col3:
                        st.metric("EPS (USD)", f"${current_eps:.2f}", 
                                 f"{eps_change:+.2f}%")
            
            # ROE
            profit_data = categorized_data.get('Profitability Ratios')
            if profit_data is not None and len(profit_data) > 0 and latest_year is not None:
                roe_row = profit_data[profit_data['Parameters'] == 'Return on Equity']
                if not roe_row.empty:
                    current_roe = float(str(roe_row[latest_year].iloc[0]).replace(',', ''))
                    
                    with col4:
                        st.metric("Return on Equity", f"{current_roe:.2f}%")
            
            st.markdown("---")
            
            # Revenue and Profit Trends
            st.subheader("Revenue and Profit Trends (2020-2024)")
            if income_data is not None:
                revenue_profit_data = income_data[income_data['Parameters'].isin(['Total Revenue', 'Net Income', 'Operating Income'])]
                fig = create_comparison_chart(revenue_profit_data, "Revenue and Profit Trends")
                st.plotly_chart(fig, use_container_width=True)
            
            # Key Insights
            st.subheader("Key Financial Insights")
            insights = analyzer.generate_executive_summary()
            for insight in insights:
                st.info(f"• {insight}")
        
        elif page == "Financial Analysis":
            st.subheader("📊 Complete Financial Analysis")
            st.markdown("Comprehensive view of all financial statements and key ratios")
            
            # Create tabs for better organization
            tab1, tab2, tab3 = st.tabs(["📈 Financial Statements", "📊 Performance Ratios", "💹 Operational Ratios"])
            
            with tab1:
                st.markdown("### Financial Statements Overview")
                financial_statement_categories = ['Income Statement', 'Balance Sheet', 'Cash Flow']
                
                for category_name in financial_statement_categories:
                    if category_name in categorized_data:
                        display_category_data(category_name, categorized_data[category_name], analyzer)
            
            with tab2:
                st.markdown("### Performance and Growth Metrics")
                performance_categories = ['Growth Ratios', 'Equity Ratios', 'Profitability Ratios']
                
                for category_name in performance_categories:
                    if category_name in categorized_data:
                        display_category_data(category_name, categorized_data[category_name], analyzer)
            
            with tab3:
                st.markdown("### Operational and Financial Health Ratios")
                operational_categories = ['Cost Ratios', 'Liquidity Ratios', 'Leverage Ratios', 'Efficiency Ratios']
                
                for category_name in operational_categories:
                    if category_name in categorized_data:
                        display_category_data(category_name, categorized_data[category_name], analyzer)
        
        elif page == "Price Analysis":
            display_price_analysis(TICKER, cached_info)
        
        elif page == "ESG Analysis":
            display_esg_analysis(TICKER, cached_info)
        
        elif page == "Industry Benchmarking":
            display_industry_benchmarking(TICKER, cached_info)
        
        elif page == "Risk Analysis":
            display_risk_analysis(TICKER, cached_info)
        
        elif page == "Custom Analysis":
            st.subheader("Custom Analysis")
            
            st.write("Select metrics to compare:")
            
            # Get all available metrics
            all_metrics = []
            for category_name, data in categorized_data.items():
                for param in data['Parameters'].tolist():
                    all_metrics.append(f"{category_name}: {param}")
            
            selected_metrics = st.multiselect(
                "Choose metrics to visualize:",
                all_metrics,
                default=all_metrics[:3] if len(all_metrics) >= 3 else all_metrics
            )
            
            if selected_metrics:
                # Create custom comparison
                custom_data_rows = []
                
                for metric in selected_metrics:
                    category, param = metric.split(': ', 1)
                    data = categorized_data[category]
                    row = data[data['Parameters'] == param]
                    if not row.empty:
                        custom_data_rows.append(row)
                
                if custom_data_rows:
                    custom_df = pd.concat(custom_data_rows, ignore_index=True)
                    
                    st.dataframe(custom_df, use_container_width=True)
                    
                    fig = create_comparison_chart(custom_df, "Custom Metrics Comparison")
                    st.plotly_chart(fig, use_container_width=True)
    
    except FileNotFoundError:
        st.error(f"❌ Data file not found at: {data_path}")
        st.info("Please make sure the file 'QCOMfinancials.csv' is in the 'data' folder.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
