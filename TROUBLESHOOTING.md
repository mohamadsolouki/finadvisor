# Yahoo Finance Connection Troubleshooting Guide

## Common Issues and Solutions

### Issue 1: "Failed to get ticker" or "No price data found"

**Possible Causes:**
- Yahoo Finance API is temporarily down or rate-limiting
- Network connectivity issues
- Firewall/proxy blocking requests
- Invalid ticker symbol
- Yahoo Finance changed their API

**Solutions:**

1. **Check Internet Connection:**
   ```powershell
   ping finance.yahoo.com
   ```

2. **Test ticker directly in Python:**
   ```powershell
   python
   ```
   ```python
   import yfinance as yf
   ticker = yf.Ticker("QCOM")
   print(ticker.history(period="1d"))
   ```

3. **Clear cache and retry:**
   - Click "Clear Cache & Refresh Data" button in sidebar
   - Wait 30 seconds
   - Try again

4. **Update yfinance:**
   ```powershell
   pip install --upgrade yfinance
   ```

5. **Try with shorter period first:**
   - Change from 5y to 1y or 1mo
   - If that works, gradually increase period

### Issue 2: Rate Limiting (429 Errors)

**Solutions:**

1. **Wait it out:**
   - Wait 10-15 minutes before trying again
   - Yahoo Finance has daily/hourly limits

2. **Use cached data:**
   - Data is cached for 2 hours
   - Avoid clearing cache frequently

3. **Reduce requests:**
   - Don't navigate between pages too quickly
   - Let pages fully load before switching

### Issue 3: JSON Decode Errors

**Causes:**
- Yahoo Finance returned HTML instead of JSON
- Server error on Yahoo's side
- API endpoint changed

**Solutions:**

1. **Retry with delays:**
   - The app now has automatic retry logic (3 attempts)
   - Each attempt waits longer

2. **Check yfinance issues:**
   - Visit: https://github.com/ranaroussi/yfinance/issues
   - Check for recent API changes

3. **Alternative data sources:**
   - Consider using Alpha Vantage
   - Use pandas_datareader
   - Use finnhub API

## Enhanced Error Handling (Already Implemented)

The app now includes:

### 1. **Retry Logic**
- Automatically retries failed requests 3 times
- Increasing delays between retries (0.5s, 1s, 1.5s)
- Graceful fallback to empty data

### 2. **Custom User-Agent**
- Uses browser-like user agent
- Helps avoid bot detection
- Better success rate

### 3. **Session Management**
- Creates persistent sessions
- Reuses connections
- Reduces overhead

### 4. **Better Error Messages**
- Clear, actionable error messages
- Suggestions for resolution
- No confusing technical jargon

## Manual Testing Steps

1. **Test Basic Connectivity:**
   ```python
   import yfinance as yf
   import requests
   
   # Test with session
   session = requests.Session()
   session.headers.update({'User-Agent': 'Mozilla/5.0'})
   
   ticker = yf.Ticker("QCOM", session=session)
   print("Info:", ticker.info.get('longName', 'No data'))
   print("History:", len(ticker.history(period="1mo")))
   ```

2. **Check if issue is specific to QCOM:**
   ```python
   # Try different tickers
   for symbol in ['AAPL', 'MSFT', 'GOOGL', 'QCOM']:
       try:
           t = yf.Ticker(symbol)
           hist = t.history(period="1d")
           print(f"{symbol}: {'OK' if not hist.empty else 'FAIL'}")
       except Exception as e:
           print(f"{symbol}: ERROR - {e}")
   ```

3. **Test with different periods:**
   ```python
   ticker = yf.Ticker("QCOM")
   for period in ['1d', '5d', '1mo', '3mo', '1y', '5y']:
       try:
           hist = ticker.history(period=period)
           print(f"{period}: {len(hist)} rows")
       except Exception as e:
           print(f"{period}: ERROR")
   ```

## Workarounds

### Option 1: Use CSV Data Only
If Yahoo Finance is completely unavailable:
1. Disable Yahoo Finance pages temporarily
2. Use only the "Financial Analysis" page with CSV data
3. Comment out price/ESG/risk pages in navigation

### Option 2: Mock Data for Testing
Create mock data for development:
```python
# In utils/data_fetcher.py, add:
@st.cache_data(ttl=7200)
def get_stock_info_mock(ticker):
    return {
        'longName': 'Qualcomm Inc.',
        'sector': 'Technology',
        'marketCap': 150000000000,
        'trailingPE': 20.5,
        # ... more fields
    }
```

### Option 3: Alternative Data Source
Consider using Alpha Vantage (free tier available):
```python
import requests

def get_alphavantage_data(ticker, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}'
    response = requests.get(url)
    return response.json()
```

## Prevention

1. **Don't clear cache frequently**
   - Cache lasts 2 hours for a reason
   - Reduces API calls significantly

2. **Batch operations**
   - Load all data at once when possible
   - Avoid rapid page switching

3. **Off-peak usage**
   - Use during off-peak hours (night/early morning)
   - Yahoo Finance has global usage patterns

4. **Monitor yfinance updates**
   - Star the repo: https://github.com/ranaroussi/yfinance
   - Watch for breaking changes
   - Update regularly

## Current Status Check

Run this diagnostic script:

```python
# test_yahoo_finance.py
import yfinance as yf
import requests
from datetime import datetime

print(f"Test run: {datetime.now()}")
print("-" * 50)

# Test 1: Basic ticker
print("\n1. Testing basic ticker fetch...")
try:
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
    
    ticker = yf.Ticker("QCOM", session=session)
    info = ticker.info
    print(f"   ✓ Company name: {info.get('longName', 'Unknown')}")
    print(f"   ✓ Market cap: ${info.get('marketCap', 0)/1e9:.2f}B")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Historical data
print("\n2. Testing historical data...")
try:
    hist = ticker.history(period="1mo")
    print(f"   ✓ Retrieved {len(hist)} days of data")
    print(f"   ✓ Latest close: ${hist['Close'].iloc[-1]:.2f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Multiple periods
print("\n3. Testing different periods...")
for period in ['1d', '5d', '1mo', '1y']:
    try:
        hist = ticker.history(period=period)
        status = "✓" if not hist.empty else "✗"
        print(f"   {status} {period}: {len(hist)} rows")
    except Exception as e:
        print(f"   ✗ {period}: Failed")

print("\n" + "-" * 50)
print("Test complete!")
```

Save as `test_yahoo_finance.py` and run:
```powershell
python test_yahoo_finance.py
```

This will show exactly where the issue is.
