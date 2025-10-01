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
        'MRVL': 'Marvell Technology',
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
            # Calculate average P/E excluding None values
            pe_series = df[df['Ticker'] != ticker]['P/E Ratio']
            avg_pe = pe_series[pe_series.notna()].mean()
            
            qcom_margin = qcom_data['Profit Margin (%)'].iloc[0]
            avg_margin = df[df['Ticker'] != ticker]['Profit Margin (%)'].mean()
            
            qcom_roe = qcom_data['ROE (%)'].iloc[0]
            avg_roe = df[df['Ticker'] != ticker]['ROE (%)'].mean()
            
            # Only show P/E comparison if we have valid data
            if qcom_pe is not None and not pd.isna(qcom_pe) and avg_pe is not None and not pd.isna(avg_pe):
                if qcom_pe < avg_pe:
                    st.info(f"📉 **{ticker} P/E Ratio** ({qcom_pe:.2f}) is below industry average ({avg_pe:.2f}), suggesting potential undervaluation")
                else:
                    st.info(f"📈 **{ticker} P/E Ratio** ({qcom_pe:.2f}) is above industry average ({avg_pe:.2f}), suggesting premium valuation")
            
            if qcom_margin > avg_margin:
                st.success(f"✅ **{ticker} Profit Margin** ({qcom_margin:.2f}%) exceeds industry average ({avg_margin:.2f}%)")
            
            if qcom_roe > avg_roe:
                st.success(f"✅ **{ticker} ROE** ({qcom_roe:.2f}%) is above industry average ({avg_roe:.2f}%)")
        
        # AI Interpretation Section
        st.markdown("---")
        st.markdown("### 🤖 AI-Powered Comprehensive Analysis")
        st.markdown("Deep dive into the competitive landscape with AI-generated insights")
        
        # Import AI insights generator
        from utils.ai_insights_generator import AIInsightsGenerator
        from pathlib import Path
        
        # Initialize AI generator
        data_dir = Path("data")
        ai_generator = AIInsightsGenerator(data_dir)
        
        if ai_generator.enabled:
            with st.spinner("🧠 Generating comprehensive AI analysis..."):
                # Prepare comprehensive context for AI
                context_data = df.copy()
                
                # Create a detailed prompt for industry benchmarking
                prompt = f"""You are a senior financial analyst providing a comprehensive competitive analysis of {ticker} in the semiconductor industry.

**Industry Benchmarking Data:**
{context_data.to_string(index=False)}

**Visual Analysis Context:**
The analysis includes four key visualizations:
1. Market Cap Comparison: Shows relative company sizes (highest to lowest market cap)
2. P/E Ratio Comparison: Indicates valuation levels and growth expectations
3. Profit Margins: Demonstrates operational efficiency and pricing power
4. ROE Comparison: Reflects how effectively companies generate returns on equity

**Your Task:**
Provide a comprehensive, insightful analysis (500-700 words) that:

1. **Competitive Position**: Analyze {ticker}'s standing relative to competitors across all metrics shown in the charts. Reference specific companies and their positions.

2. **Valuation Analysis**: Interpret the P/E ratio patterns. What does {ticker}'s valuation suggest about market expectations? Compare with Broadcom, AMD, and other peers shown in the P/E chart.

3. **Profitability & Efficiency**: Deep dive into the profit margin and ROE charts. Which companies show operational excellence? How does {ticker} compare? What might explain the differences?

4. **Market Cap Insights**: Analyze the market cap distribution shown in the chart. What does company size suggest about competitive advantages, market share, or investor confidence?

5. **Strategic Implications**: Based on ALL the data and charts presented, what strategic position does {ticker} occupy? Is it a value play, growth story, or quality compounder?

6. **Investment Perspective**: Synthesize all metrics to provide a balanced view for investors. Which metrics are most favorable/concerning for {ticker}?

7. **Industry Trends**: What do these collective metrics reveal about the semiconductor industry competitive dynamics?

Be specific, cite actual numbers from the data, reference the visualizations, and provide actionable insights. Write in a professional yet accessible tone suitable for both institutional and retail investors."""
                
                try:
                    # Generate AI insight using OpenAI directly with custom prompt
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
                                "content": "You are a senior financial analyst with deep expertise in the semiconductor industry. Provide comprehensive, data-driven insights that reference specific metrics and visualizations."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    ai_insight = response.choices[0].message.content.strip()
                    
                    # Display AI insight in an attractive format
                    st.markdown("""<div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 5px solid #1f77b4;"></div>""", unsafe_allow_html=True)
                    st.markdown(ai_insight)
                    
                    # Add disclaimer
                    st.caption("💡 AI-generated analysis based on current market data. This analysis references all charts and metrics shown above. Always conduct additional research before making investment decisions.")
                    
                except Exception as e:
                    st.warning(f"⚠️ Unable to generate AI analysis: {str(e)}")
                    st.info("Please ensure your OpenAI API key is properly configured in the .env file.")
        else:
            st.info("🔑 **AI Analysis Unavailable**: Configure your OpenAI API key in the .env file to enable comprehensive AI-powered insights that analyze all the data and charts on this page.")
    else:
        st.error("⚠️ Unable to load competitor data. Please refresh or try again later.")
        st.info("💡 Tip: Check your internet connection and ensure Yahoo Finance is accessible.")
