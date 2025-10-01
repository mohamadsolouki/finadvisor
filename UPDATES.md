# Financial Advisor Application - Recent Updates

## Overview
The Financial Advisor application has been restructured to provide a comprehensive financial analysis dashboard with integrated Yahoo Finance data and improved performance through intelligent caching.

## Major Changes

### 1. Restructured Navigation
**Previous Structure:**
- Multiple separate pages for each financial category

**New Structure:**
- **Executive Summary** - Key metrics and insights at a glance
- **Financial Analysis** - Main integrated page with all financial statements and ratios organized in tabs
- **Price Analysis** - Stock price charts, technical indicators, and trading analysis
- **ESG Analysis** - Environmental, Social, and Governance metrics
- **Industry Benchmarking** - Peer comparison with competitors
- **Risk Analysis** - Risk metrics, volatility analysis, and portfolio statistics
- **Custom Analysis** - User-selected metrics comparison

### 2. Financial Analysis Page (Main Page)
The Financial Analysis page now consolidates all financial data into a single, well-organized view with three tabs:

#### 📈 Financial Statements Tab
- Income Statement
- Balance Sheet
- Cash Flow

#### 📊 Performance Ratios Tab
- Growth Ratios
- Equity Ratios
- Profitability Ratios

#### 💹 Operational Ratios Tab
- Cost Ratios
- Liquidity Ratios
- Leverage Ratios
- Efficiency Ratios

### 3. Enhanced Yahoo Finance Integration

#### Price Analysis Features
- **Current Price Metrics**: Real-time price, 52-week high/low, volume
- **Technical Indicators**:
  - Candlestick charts
  - Moving Averages (20-day, 50-day, 200-day)
  - Bollinger Bands
  - Relative Strength Index (RSI)
- **Performance Metrics**: 1M, 3M, 1Y, YTD returns
- **Volume Analysis**: Trading volume trends
- **Returns Distribution**: Statistical analysis of daily returns

#### ESG Analysis Features
- Total ESG Score
- Environment, Social, and Governance component scores
- Visual breakdown charts
- Detailed ESG metrics table
- Company context (sector, industry, employees)
- Controversy levels (when available)

#### Industry Benchmarking Features
- Company overview metrics
- Profitability and efficiency metrics
- **Peer Comparison** with major semiconductor companies:
  - Qualcomm (QCOM)
  - NVIDIA (NVDA)
  - Broadcom (AVGO)
  - Texas Instruments (TXN)
  - AMD (AMD)
  - Intel (INTC)
- Visual comparisons: Market Cap, Profit Margin, ROE, P/E Ratio
- Side-by-side metric tables

#### Risk Analysis Features
- **Key Risk Metrics**:
  - Beta (market volatility comparison)
  - Annual Volatility
  - Sharpe Ratio (risk-adjusted returns)
  - Maximum Drawdown
  - Downside Deviation
  - Sortino Ratio
  - Gain/Loss Ratio
- **Value at Risk (VaR)** at 90%, 95%, and 99% confidence levels
- **Expected Shortfall (CVaR)** - worst-case scenario analysis
- **Rolling Volatility Charts** - 30-day rolling volatility
- **Returns Distribution** - histogram with VaR markers
- **Drawdown Analysis** - historical drawdown periods and recovery times
- **Risk Interpretations** - contextual explanations of metrics

### 4. Performance Optimization

#### Caching Strategy
To avoid Yahoo Finance API rate limiting (429 errors), we implemented aggressive caching:

- **Cache Duration**: 2 hours (7200 seconds)
- **Cached Functions**:
  - `get_stock_info()` - Company information
  - `get_historical_data()` - Price history
  - `get_esg_data()` - ESG metrics
  - `get_competitor_data()` - Peer comparison data

#### Rate Limiting Prevention
- **Delays between API calls**: 200-500ms to avoid rapid consecutive requests
- **Single data fetch per page**: Data fetched once and passed to functions
- **Session state tracking**: Monitors last fetch time
- **User feedback**: Clear messages when rate limits are reached

#### Data Management
- **Manual refresh button**: "Clear Cache & Refresh Data" in sidebar
- **Last refresh timestamp**: Shows time since last data update
- **Error handling**: Graceful degradation with helpful user messages

### 5. User Experience Improvements

#### Visual Enhancements
- Color-coded metrics (green for positive, red for negative)
- Interactive charts with hover details
- Tabbed layouts for better organization
- Highlighted target company in peer comparisons
- Clear section headers with emojis

#### Error Handling
- Informative error messages
- Rate limiting warnings
- Tips for troubleshooting
- Graceful fallbacks when data unavailable

#### Information Architecture
- Sidebar with navigation and data management
- Context-sensitive info boxes
- Performance tips in sidebar
- Data source transparency

## Technical Details

### Data Sources
1. **CSV File**: `data/QCOMfinancials.csv`
   - Historical financial statements
   - Financial ratios
   - Multi-year comparisons

2. **Yahoo Finance API**:
   - Real-time stock prices
   - Historical price data
   - Company information
   - ESG scores
   - Peer company data

### Caching Architecture
```python
@st.cache_data(ttl=7200)  # 2-hour cache
def get_stock_info(ticker):
    time.sleep(0.2)  # Rate limiting
    stock = yf.Ticker(ticker)
    return stock.info
```

### Session State Management
```python
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = None
```

## Usage Guidelines

### Best Practices
1. **Avoid frequent refreshes**: Data is cached for 2 hours - use it!
2. **Wait between refreshes**: If you clear cache, wait 30 seconds before navigating
3. **Handle rate limits gracefully**: If you see 429 errors, wait 5-10 minutes
4. **Use during off-peak hours**: Yahoo Finance has usage limits

### Troubleshooting

#### "429 Too Many Requests" Error
- **Cause**: Yahoo Finance rate limiting
- **Solution**: Wait 5-10 minutes, then click "Clear Cache & Refresh Data"

#### Data Not Loading
- **Check internet connection**
- **Verify Yahoo Finance is accessible**
- **Clear cache and wait before refreshing**

#### Outdated Data
- Use "Clear Cache & Refresh Data" button in sidebar
- Note: Only refresh when necessary to avoid rate limits

## Future Enhancements

### Potential Additions
1. **More data sources**: Integration with additional financial APIs
2. **Custom ticker selection**: Allow users to analyze different companies
3. **Historical ESG tracking**: Track ESG scores over time
4. **Advanced technical indicators**: MACD, Stochastic, etc.
5. **Portfolio analysis**: Multi-stock portfolio tracking
6. **Export enhanced data**: Include Yahoo Finance data in reports
7. **Alerts and notifications**: Price and metric alerts
8. **Comparison tool**: Compare multiple companies side-by-side

### Performance Improvements
1. **Background data refresh**: Automatic cache warming
2. **Progressive loading**: Load critical data first
3. **Local database**: Store historical data locally
4. **Compressed caching**: Reduce memory footprint

## Dependencies

Required packages (in `requirements.txt`):
```
streamlit==1.31.0
pandas==2.2.0
plotly==5.18.0
reportlab==4.0.9
openpyxl==3.1.2
numpy==1.26.3
yfinance==0.2.28
```

## Running the Application

1. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

2. **Run the app**:
   ```powershell
   streamlit run app.py
   ```

3. **Access in browser**: Opens automatically at `http://localhost:8501`

## Support

For issues or questions:
- Check `QUICKSTART.md` for setup instructions
- Review `README.md` for project overview
- Check this document for recent changes
