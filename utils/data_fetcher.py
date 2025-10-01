"""
Data fetching module for Yahoo Finance integration
Handles all API calls with caching to avoid rate limiting
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import time
import json


@st.cache_data(ttl=7200, show_spinner=False)  # Cache for 2 hours to reduce API calls
def get_stock_data(ticker, period="5y"):
    """Fetch stock data from Yahoo Finance with caching and retry logic"""
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Add delay between retries
            if attempt > 0:
                time.sleep(1.5 * attempt)
            
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Fetch historical data
            hist = stock.history(period=period)
            
            if hist.empty:
                last_error = f"No historical data returned for period={period}"
                if attempt < max_retries - 1:
                    continue
                else:
                    st.error(f"⚠️ No price data found for {ticker}. The ticker may be invalid or data is temporarily unavailable.")
                    return None, pd.DataFrame(), {}
            
            # Fetch info - catch JSON decode errors
            try:
                info = stock.info
                
                # Validate info is not empty
                if not info or len(info) < 5:
                    last_error = "Info dictionary is empty or too small"
                    if attempt < max_retries - 1:
                        continue
                    else:
                        # Return with historical data but empty info
                        return stock, hist, {}
                
                return stock, hist, info
                
            except (json.JSONDecodeError, ValueError) as json_err:
                last_error = f"JSON decode error: {str(json_err)}"
                if attempt < max_retries - 1:
                    continue
                else:
                    # Return with historical data but empty info
                    return stock, hist, {}
            
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                continue
            else:
                st.error(f"⚠️ Error fetching data for {ticker}: {str(e)}")
                st.info("💡 Tips:\n- Check your internet connection\n- Clear cache (C key in sidebar)\n- Wait a few minutes if rate-limited")
                return None, pd.DataFrame(), {}
    
    if last_error:
        st.warning(f"Fetch completed with warnings: {last_error}")
    
    return None, pd.DataFrame(), {}


@st.cache_data(ttl=7200, show_spinner=False)  # Cache for 2 hours
def get_stock_info(ticker):
    """Fetch only stock info with retry logic"""
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Add delay between retries
            if attempt > 0:
                time.sleep(1.5 * attempt)
            
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Fetch info - catch JSON decode errors
            try:
                info = stock.info
                
                # Check if info is valid
                if not info or len(info) < 5:
                    last_error = "Info dictionary is empty or too small"
                    if attempt < max_retries - 1:
                        continue
                    else:
                        st.warning(f"⚠️ Limited data available for {ticker}")
                        return {}
                
                return info
                
            except (json.JSONDecodeError, ValueError) as json_err:
                last_error = f"JSON decode error: {str(json_err)}"
                if attempt < max_retries - 1:
                    continue
                else:
                    st.warning(f"⚠️ Data format error for {ticker}. Trying alternative method...")
                    return {}
            
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                continue
            else:
                st.warning(f"⚠️ Unable to fetch market data for {ticker}")
                if "429" in str(e) or "Too Many Requests" in str(e):
                    st.error("🚫 **Rate Limit**: Wait 5-10 minutes before retrying")
                elif "Expecting value" in str(e):
                    st.info("💡 Yahoo Finance returned invalid data. Try refreshing the page or clearing cache.")
                return {}
    
    if last_error:
        st.warning(f"Info fetch completed with warnings: {last_error}")
    
    return {}


@st.cache_data(ttl=7200, show_spinner=False)  # Cache for 2 hours
def get_historical_data(ticker, period="5y"):
    """Fetch only historical price data with retry logic"""
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Add delay between retries
            if attempt > 0:
                time.sleep(1.5 * attempt)
            
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Fetch historical data
            hist = stock.history(period=period)
            
            if hist.empty:
                last_error = f"No data for period={period}"
                if attempt < max_retries - 1:
                    continue
                else:
                    st.warning(f"⚠️ No historical price data found for {ticker}")
                    return pd.DataFrame()
            
            return hist
            
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                continue
            else:
                st.warning(f"⚠️ Unable to fetch historical price data.")
                if "429" in str(e) or "Too Many Requests" in str(e):
                    st.error("🚫 **Rate Limit**: Wait 5-10 minutes")
                elif "Expecting value" in str(e):
                    st.info("💡 Yahoo Finance returned invalid data. Try refreshing the page or clearing cache.")
                return pd.DataFrame()
    
    if last_error:
        st.warning(f"Historical data fetch completed with warnings: {last_error}")
    
    return pd.DataFrame()


@st.cache_data(ttl=7200)  # Cache for 2 hours
def get_esg_data(ticker):
    """Fetch ESG data separately with caching and retry logic"""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            time.sleep(0.5 * (attempt + 1))
            
            # Let yfinance handle session management (no custom session)
            stock = yf.Ticker(ticker)
            esg_data = stock.sustainability
            return esg_data
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
    
    return None


@st.cache_data(ttl=7200)  # Cache for 2 hours
def get_competitor_data(ticker_list):
    """Fetch competitor data in batch with caching and rate limiting"""
    comparison_data = []
    
    for idx, (ticker_sym, name) in enumerate(ticker_list.items()):
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Add delay between requests to avoid rate limiting
                time.sleep(0.5 * (idx + 1) + (0.3 * attempt))
                
                # Let yfinance handle session management (no custom session)
                comp_stock = yf.Ticker(ticker_sym)
                comp_info = comp_stock.info
                
                # Validate we got data
                if not comp_info or len(comp_info) < 5:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        break
                
                # Handle P/E ratio - use forwardPE if trailingPE is None or negative
                trailing_pe = comp_info.get('trailingPE')
                if trailing_pe is None or trailing_pe <= 0:
                    trailing_pe = comp_info.get('forwardPE', None)
                
                comparison_data.append({
                    'Company': name,
                    'Ticker': ticker_sym,
                    'Market Cap (B)': comp_info.get('marketCap', 0) / 1e9,
                    'P/E Ratio': trailing_pe if trailing_pe and trailing_pe > 0 else None,
                    'P/B Ratio': comp_info.get('priceToBook', 0),
                    'Profit Margin (%)': comp_info.get('profitMargins', 0) * 100,
                    'Operating Margin (%)': comp_info.get('operatingMargins', 0) * 100,
                    'ROE (%)': comp_info.get('returnOnEquity', 0) * 100,
                    'Debt/Equity': comp_info.get('debtToEquity', 0),
                    'Dividend Yield (%)': comp_info.get('dividendYield', 0) * 100 if comp_info.get('dividendYield') else 0
                })
                break  # Success, move to next ticker
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                # Skip failed tickers silently after retries
                break
    
    return comparison_data


@st.cache_data(ttl=7200)  # Cache for 2 hours
def get_competitor_esg_data(ticker_list):
    """Fetch ESG data for competitors with caching and rate limiting"""
    esg_comparison_data = []
    
    for idx, (ticker_sym, name) in enumerate(ticker_list.items()):
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Add delay between requests to avoid rate limiting
                time.sleep(0.5 * (idx + 1) + (0.3 * attempt))
                
                # Let yfinance handle session management
                comp_stock = yf.Ticker(ticker_sym)
                esg_data = comp_stock.sustainability
                
                # Check if ESG data is available
                if esg_data is not None and not esg_data.empty:
                    total_esg = esg_data.loc['totalEsg'].iloc[0] if 'totalEsg' in esg_data.index else None
                    env_score = esg_data.loc['environmentScore'].iloc[0] if 'environmentScore' in esg_data.index else None
                    social_score = esg_data.loc['socialScore'].iloc[0] if 'socialScore' in esg_data.index else None
                    gov_score = esg_data.loc['governanceScore'].iloc[0] if 'governanceScore' in esg_data.index else None
                    
                    esg_comparison_data.append({
                        'Company': name,
                        'Ticker': ticker_sym,
                        'Total ESG Score': total_esg,
                        'Environment Score': env_score,
                        'Social Score': social_score,
                        'Governance Score': gov_score
                    })
                    break  # Success
                else:
                    # No ESG data available
                    esg_comparison_data.append({
                        'Company': name,
                        'Ticker': ticker_sym,
                        'Total ESG Score': None,
                        'Environment Score': None,
                        'Social Score': None,
                        'Governance Score': None
                    })
                    break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                # Add entry with None values if all retries failed
                esg_comparison_data.append({
                    'Company': name,
                    'Ticker': ticker_sym,
                    'Total ESG Score': None,
                    'Environment Score': None,
                    'Social Score': None,
                    'Governance Score': None
                })
                break
    
    return esg_comparison_data
