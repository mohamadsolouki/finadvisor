"""
Risk Analysis Page Module
Displays volatility, VaR, CVaR, and other risk metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_fetcher import get_historical_data
from utils.visualizations import create_rolling_volatility_chart, create_drawdown_chart


def display_risk_analysis(ticker, cached_info):
    """Display risk analysis page"""
    st.subheader("⚠️ Risk Analysis")
    st.markdown(f"Comprehensive risk assessment for **{ticker}**")
    
    # Period selector
    period = st.selectbox(
        "Select Analysis Period",
        ["1y", "2y", "3y", "5y"],
        index=3,  # Default to 5y
        key="risk_period"
    )
    
    # Fetch historical data
    with st.spinner(f"Loading {period} of historical data..."):
        hist = get_historical_data(ticker, period=period)
    
    if hist.empty:
        st.error("⚠️ Unable to load historical data for risk analysis")
        st.info("💡 Try refreshing the page or selecting a different time period")
        return
    
    # Calculate returns
    hist['Returns'] = hist['Close'].pct_change()
    hist = hist.dropna()
    
    # Risk Metrics
    st.markdown("### 📊 Key Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Daily Volatility
    daily_vol = hist['Returns'].std()
    with col1:
        st.metric("Daily Volatility", f"{daily_vol*100:.2f}%")
    
    # Annualized Volatility
    annual_vol = daily_vol * np.sqrt(252)
    with col2:
        st.metric("Annual Volatility", f"{annual_vol*100:.2f}%")
    
    # Value at Risk (95%)
    var_95 = np.percentile(hist['Returns'], 5)
    with col3:
        st.metric("VaR (95%)", f"{var_95*100:.2f}%", help="95% confidence - Maximum expected loss")
    
    # Conditional VaR (CVaR)
    cvar_95 = hist['Returns'][hist['Returns'] <= var_95].mean()
    with col4:
        st.metric("CVaR (95%)", f"{cvar_95*100:.2f}%", help="Expected loss when VaR threshold is breached")
    
    st.markdown("---")
    
    # Beta and Sharpe Ratio
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Market Sensitivity")
        
        # Fetch S&P 500 for beta calculation
        try:
            from utils.data_fetcher import get_historical_data as get_sp500_data
            sp500 = get_sp500_data("^GSPC", period=period)
            
            if not sp500.empty:
                sp500['Returns'] = sp500['Close'].pct_change()
                
                # Align dates
                combined = pd.merge(
                    hist[['Returns']].rename(columns={'Returns': 'Stock'}),
                    sp500[['Returns']].rename(columns={'Returns': 'Market'}),
                    left_index=True,
                    right_index=True,
                    how='inner'
                )
                
                # Calculate Beta
                covariance = combined.cov().loc['Stock', 'Market']
                market_var = combined['Market'].var()
                beta = covariance / market_var
                
                st.metric("Beta (vs S&P 500)", f"{beta:.2f}", 
                         help="Beta > 1: More volatile than market, Beta < 1: Less volatile")
                
                if beta > 1:
                    st.info(f"📊 **{ticker}** is {beta:.2f}x as volatile as the S&P 500")
                elif beta < 1:
                    st.info(f"📊 **{ticker}** is {1/beta:.2f}x less volatile than the S&P 500")
            else:
                st.warning("Unable to calculate Beta - S&P 500 data unavailable")
        except Exception as e:
            st.warning(f"Unable to calculate Beta: {str(e)}")
    
    with col2:
        st.markdown("### 💎 Risk-Adjusted Return")
        
        # Sharpe Ratio (assuming 4% risk-free rate)
        risk_free_rate = 0.04
        avg_return = hist['Returns'].mean() * 252  # Annualized
        sharpe_ratio = (avg_return - risk_free_rate) / annual_vol
        
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}",
                 help="Higher is better - measures return per unit of risk")
        
        if sharpe_ratio > 1:
            st.success(f"✅ Good risk-adjusted returns (Sharpe > 1)")
        elif sharpe_ratio > 0:
            st.info(f"📊 Moderate risk-adjusted returns")
        else:
            st.warning(f"⚠️ Poor risk-adjusted returns (Sharpe < 0)")
    
    st.markdown("---")
    
    # Volatility Chart
    st.markdown("### 📉 Rolling Volatility (30-Day Window)")
    
    # Calculate 30-day rolling volatility
    hist['Rolling_Vol'] = hist['Returns'].rolling(window=30).std() * np.sqrt(252) * 100
    
    fig = create_rolling_volatility_chart(hist)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Drawdown Analysis
    st.markdown("### 📊 Drawdown Analysis")
    
    # Calculate cumulative returns and drawdown
    hist['Cumulative'] = (1 + hist['Returns']).cumprod()
    hist['Running_Max'] = hist['Cumulative'].expanding().max()
    hist['Drawdown'] = (hist['Cumulative'] / hist['Running_Max'] - 1) * 100
    
    max_drawdown = hist['Drawdown'].min()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%",
                 help="Largest peak-to-trough decline")
        
        # Find max drawdown date
        max_dd_date = hist['Drawdown'].idxmin()
        st.caption(f"Occurred: {max_dd_date.strftime('%Y-%m-%d')}")
    
    with col2:
        fig = create_drawdown_chart(hist)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Risk Summary
    st.markdown("### 💡 Risk Assessment Summary")
    
    risk_level = "Low"
    if annual_vol > 0.40:
        risk_level = "Very High"
        st.error(f"⚠️ **Very High Risk** - Annualized volatility of {annual_vol*100:.1f}% indicates significant price swings")
    elif annual_vol > 0.30:
        risk_level = "High"
        st.warning(f"⚠️ **High Risk** - Annualized volatility of {annual_vol*100:.1f}% suggests considerable uncertainty")
    elif annual_vol > 0.20:
        risk_level = "Moderate"
        st.info(f"📊 **Moderate Risk** - Annualized volatility of {annual_vol*100:.1f}% is typical for growth stocks")
    else:
        risk_level = "Low"
        st.success(f"✅ **Low Risk** - Annualized volatility of {annual_vol*100:.1f}% indicates relative stability")
    
    # VaR interpretation
    st.info(f"📉 **Value at Risk**: With 95% confidence, daily losses should not exceed {abs(var_95)*100:.2f}%")
    
    # Sharpe interpretation
    if sharpe_ratio > 1:
        st.success(f"📈 **Strong Risk-Adjusted Performance**: Sharpe Ratio of {sharpe_ratio:.2f} indicates good returns relative to risk")
    elif sharpe_ratio > 0:
        st.info(f"📊 **Acceptable Risk-Adjusted Performance**: Sharpe Ratio of {sharpe_ratio:.2f} is positive but moderate")
    else:
        st.warning(f"⚠️ **Poor Risk-Adjusted Performance**: Negative Sharpe Ratio of {sharpe_ratio:.2f} suggests returns don't justify the risk")
