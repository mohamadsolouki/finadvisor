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
        st.dataframe(df, width='stretch')
        
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
        from utils.text_utils import normalize_markdown_spacing
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
Provide a comprehensive, narrative-style competitive analysis (500-700 words) in a flowing, paragraph format.

Write your analysis as a cohesive competitive narrative that:

1. **Opens with Competitive Standing**: Begin by establishing {ticker}'s overall position in the semiconductor competitive landscape, referencing specific companies and their positions across the key metrics shown in the charts.

2. **Tells the Valuation Story**: Flow naturally into the valuation discussion by interpreting the P/E ratio patterns. Weave together what {ticker}'s valuation suggests about market expectations compared with Broadcom, AMD, and other peers, connecting this to broader market sentiment and growth expectations.

3. **Analyzes Operational Performance**: Transition smoothly into profitability and efficiency by discussing the profit margin and ROE charts. Tell the story of which companies demonstrate operational excellence, how {ticker} compares, and what factors might explain the differences. Connect operational metrics to competitive advantages.

4. **Provides Market Context**: Naturally incorporate insights from the market cap distribution chart, discussing what company size reveals about competitive positioning, market share dynamics, and investor confidence across the peer group.

5. **Synthesizes Strategic Position**: Throughout the narrative, build toward a synthesis of what strategic position {ticker} occupies—whether it's a value play, growth story, or quality compounder. Discuss industry trends revealed by the collective metrics.

6. **Concludes with Investment Perspective**: End with a balanced synthesis paragraph that ties together all the metrics to provide clear investment perspective, highlighting which metrics are most favorable or concerning for {ticker}.

**Formatting Requirements:**
- Write in flowing paragraphs (4-6 paragraphs total), NOT bullet points or multiple heading sections
- DO NOT use multiple heading levels (###) within your response - write as continuous prose
- Bold key figures and metrics to make them stand out: **$150.25B**, **P/E: 25.3**, **ROE: 18.5%**
- Use smooth transitions between paragraphs to maintain narrative flow
- Write in an engaging, professional tone as if briefing an investor

Be specific, cite actual numbers from the data, reference the visualizations naturally within the narrative, and provide actionable insights."""
                
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
                                "content": "You are a senior financial analyst with deep expertise in the semiconductor industry. Write in flowing, paragraph-based narrative format, NOT as bullet points or multiple sections with headings. Provide comprehensive, data-driven insights that reference specific metrics and visualizations naturally within the narrative. Tell the story of the competitive landscape as a cohesive narrative with smooth transitions between ideas."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.7,
                        max_tokens=1500
                    )
                    
                    ai_insight = normalize_markdown_spacing(response.choices[0].message.content.strip())
                    
                    # Display AI insight using Streamlit's native components
                    st.markdown("---")
                    st.info("📊 AI-Generated Competitive Analysis")
                    st.markdown(ai_insight)
                    st.markdown("---")
                    
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
