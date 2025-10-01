"""
Industry Benchmarking Page Module
Displays competitor comparison and industry metrics
"""

import streamlit as st
import pandas as pd
from utils.data_fetcher import get_competitor_data


def display_industry_benchmarking(ticker, cached_info):
    """Display industry benchmarking page"""
    st.subheader("🏢 Industry Benchmarking")
    st.markdown(f"Compare **{ticker}** with key semiconductor competitors")
    
    # Competitor list
    competitors = {
        'QCOM': 'Qualcomm',
        'AVGO': 'Broadcom',
        'NVDA': 'NVIDIA',
        'AMD': 'AMD',
        'INTC': 'Intel',
        'TXN': 'Texas Instruments'
    }
    
    # Fetch competitor data
    with st.spinner("Loading competitor data..."):
        comparison_data = get_competitor_data(competitors)
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Display data table
        st.markdown("### 📊 Financial Comparison")
        st.dataframe(df, use_container_width=True)
        
        # Key metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Market Cap Comparison")
            fig_data = df[['Company', 'Market Cap (B)']].sort_values('Market Cap (B)', ascending=False)
            st.bar_chart(fig_data.set_index('Company')['Market Cap (B)'])
        
        with col2:
            st.markdown("#### P/E Ratio Comparison")
            fig_data = df[['Company', 'P/E Ratio']].sort_values('P/E Ratio', ascending=False)
            st.bar_chart(fig_data.set_index('Company')['P/E Ratio'])
        
        # Profitability comparison
        st.markdown("### 💰 Profitability Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Profit Margins")
            fig_data = df[['Company', 'Profit Margin (%)']].sort_values('Profit Margin (%)', ascending=False)
            st.bar_chart(fig_data.set_index('Company')['Profit Margin (%)'])
        
        with col2:
            st.markdown("#### ROE Comparison")
            fig_data = df[['Company', 'ROE (%)']].sort_values('ROE (%)', ascending=False)
            st.bar_chart(fig_data.set_index('Company')['ROE (%)'])
        
        # Insights
        qcom_data = df[df['Ticker'] == ticker]
        if not qcom_data.empty:
            st.markdown("### 💡 Key Insights")
            
            qcom_pe = qcom_data['P/E Ratio'].iloc[0]
            avg_pe = df[df['Ticker'] != ticker]['P/E Ratio'].mean()
            
            qcom_margin = qcom_data['Profit Margin (%)'].iloc[0]
            avg_margin = df[df['Ticker'] != ticker]['Profit Margin (%)'].mean()
            
            qcom_roe = qcom_data['ROE (%)'].iloc[0]
            avg_roe = df[df['Ticker'] != ticker]['ROE (%)'].mean()
            
            if qcom_pe < avg_pe:
                st.info(f"📉 **{ticker} P/E Ratio** ({qcom_pe:.2f}) is below industry average ({avg_pe:.2f}), suggesting potential undervaluation")
            else:
                st.info(f"📈 **{ticker} P/E Ratio** ({qcom_pe:.2f}) is above industry average ({avg_pe:.2f}), suggesting premium valuation")
            
            if qcom_margin > avg_margin:
                st.success(f"✅ **{ticker} Profit Margin** ({qcom_margin:.2f}%) exceeds industry average ({avg_margin:.2f}%)")
            
            if qcom_roe > avg_roe:
                st.success(f"✅ **{ticker} ROE** ({qcom_roe:.2f}%) is above industry average ({avg_roe:.2f}%)")
    else:
        st.error("⚠️ Unable to load competitor data. Please refresh or try again later.")
        st.info("💡 Tip: Check your internet connection and ensure Yahoo Finance is accessible.")
