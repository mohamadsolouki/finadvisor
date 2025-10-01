"""
Data fetching module for Yahoo Finance integration
Handles all API calls with caching to avoid rate limiting
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import time
import requests


@st.cache_data(ttl=7200)  # Cache for 2 hours to reduce API calls
def get_stock_data(ticker, period="5y"):
    """Fetch stock data from Yahoo Finance with caching and retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(0.5 * (attempt + 1))  # Increasing delay with each retry
            
            # Create ticker with session
            session = requests.Session()
            session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
            
            stock = yf.Ticker(ticker, session=session)
            hist = stock.history(period=period)
            
            if hist.empty:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    st.error(f"⚠️ No price data found for {ticker}. The ticker may be invalid or data is temporarily unavailable.")
                    return None, pd.DataFrame(), {}
            
            info = stock.info
            return stock, hist, info
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                st.error(f"⚠️ Error fetching data for {ticker}: {str(e)}")
                st.info("💡 Tips:\n- Check your internet connection\n- Clear cache and try again\n- Wait a few minutes if rate-limited")
                return None, pd.DataFrame(), {}
    
    return None, pd.DataFrame(), {}


@st.cache_data(ttl=7200)  # Cache for 2 hours
def get_stock_info(ticker):
    """Fetch only stock info with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(0.5 * (attempt + 1))
            
            # Create ticker with session
            session = requests.Session()
            session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
            
            stock = yf.Ticker(ticker, session=session)
            info = stock.info
            
            # Check if info is valid
            if not info or len(info) < 5:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    st.warning(f"⚠️ Limited data available for {ticker}")
                    return {}
            
            return info
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                st.warning(f"⚠️ Unable to fetch market data for {ticker}")
                if "429" in str(e) or "Too Many Requests" in str(e):
                    st.error("🚫 **Rate Limit**: Wait 5-10 minutes before retrying")
                return {}
    
    return {}


@st.cache_data(ttl=7200)  # Cache for 2 hours
def get_historical_data(ticker, period="5y"):
    """Fetch only historical price data with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(0.5 * (attempt + 1))
            
            # Create ticker with session
            session = requests.Session()
            session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
            
            stock = yf.Ticker(ticker, session=session)
            hist = stock.history(period=period)
            
            if hist.empty:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    st.warning(f"⚠️ No historical price data found for {ticker}")
                    return pd.DataFrame()
            
            return hist
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            else:
                st.warning(f"⚠️ Unable to fetch historical price data.")
                if "429" in str(e) or "Too Many Requests" in str(e):
                    st.error("🚫 **Rate Limit**: Wait 5-10 minutes")
                return pd.DataFrame()
    
    return pd.DataFrame()


@st.cache_data(ttl=7200)  # Cache for 2 hours
def get_esg_data(ticker):
    """Fetch ESG data separately with caching and retry logic"""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            time.sleep(0.5 * (attempt + 1))
            
            session = requests.Session()
            session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
            
            stock = yf.Ticker(ticker, session=session)
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
                
                session = requests.Session()
                session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
                
                comp_stock = yf.Ticker(ticker_sym, session=session)
                comp_info = comp_stock.info
                
                # Validate we got data
                if not comp_info or len(comp_info) < 5:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        break
                
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
                break  # Success, move to next ticker
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                # Skip failed tickers silently after retries
                break
    
    return comparison_data
