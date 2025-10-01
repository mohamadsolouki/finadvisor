"""
ESG Analysis Page Module
Displays Environmental, Social, and Governance metrics
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_fetcher import get_stock_info, get_esg_data


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
                **ESG Risk Scores:**
                
                Lower is better:
                - **0-10**: Negligible Risk
                - **10-20**: Low Risk
                - **20-30**: Medium Risk
                - **30-40**: High Risk
                - **40+**: Severe Risk
                
                Scores measure ESG risk exposure and management quality.
                """)
            
            st.markdown("---")
            
            # Display full ESG dataframe
            st.subheader("📋 Detailed ESG Metrics")
            st.dataframe(esg_data, use_container_width=True)
            
            # Competitor ESG Comparison
            st.markdown("---")
            st.subheader("🏢 Industry ESG Comparison")
            st.markdown(f"Compare **{ticker}** ESG scores with key semiconductor competitors")
            
            # Competitor list (same as industry benchmarking)
            competitors = {
                'QCOM': 'Qualcomm',
                'AVGO': 'Broadcom',
                'MRVL': 'Marvell Technology',
                'AMD': 'AMD',
                'INTC': 'Intel',
                'TXN': 'Texas Instruments'
            }
            
            # Import the competitor ESG data function
            from utils.data_fetcher import get_competitor_esg_data
            
            with st.spinner("Loading competitor ESG data..."):
                esg_comparison = get_competitor_esg_data(competitors)
            
            if esg_comparison:
                esg_df = pd.DataFrame(esg_comparison)
                
                # Check if we have any valid ESG data
                valid_data = esg_df[esg_df['Total ESG Score'].notna()]
                
                if not valid_data.empty:
                    # Display comparison table
                    st.dataframe(esg_df, use_container_width=True)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Total ESG Score Comparison")
                        # Filter out None values for visualization
                        plot_data = esg_df[esg_df['Total ESG Score'].notna()]
                        if not plot_data.empty:
                            fig_data = plot_data[['Company', 'Total ESG Score']].sort_values('Total ESG Score')
                            st.bar_chart(fig_data.set_index('Company')['Total ESG Score'])
                        else:
                            st.info("No ESG scores available for visualization")
                    
                    with col2:
                        st.markdown("#### ESG Component Scores")
                        # Show average scores for each component
                        component_avg = {
                            'Environment': esg_df['Environment Score'].mean(skipna=True),
                            'Social': esg_df['Social Score'].mean(skipna=True),
                            'Governance': esg_df['Governance Score'].mean(skipna=True)
                        }
                        if not all(pd.isna(v) for v in component_avg.values()):
                            st.bar_chart(pd.DataFrame(component_avg, index=['Industry Avg']).T)
                        else:
                            st.info("No component scores available")
                    
                    # Insights
                    qcom_esg = esg_df[esg_df['Ticker'] == ticker]
                    if not qcom_esg.empty and qcom_esg['Total ESG Score'].iloc[0] is not None:
                        qcom_score = qcom_esg['Total ESG Score'].iloc[0]
                        avg_score = esg_df[esg_df['Ticker'] != ticker]['Total ESG Score'].mean(skipna=True)
                        
                        st.markdown("### 💡 ESG Insights")
                        if not pd.isna(avg_score):
                            if qcom_score < avg_score:
                                st.success(f"✅ **{ticker} ESG Risk Score** ({qcom_score:.2f}) is lower than industry average ({avg_score:.2f}), indicating better ESG risk management")
                            else:
                                st.warning(f"⚠️ **{ticker} ESG Risk Score** ({qcom_score:.2f}) is higher than industry average ({avg_score:.2f}), indicating higher ESG risk exposure")
                else:
                    st.info("ESG data is not available for most companies in this comparison.")
            
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
            
            # AI Interpretation Section for ESG
            st.markdown("---")
            st.markdown("### 🤖 AI-Powered ESG Analysis")
            st.markdown("Comprehensive interpretation of ESG performance and competitive positioning")
            
            # Import AI insights generator
            from utils.ai_insights_generator import AIInsightsGenerator
            from pathlib import Path
            
            # Initialize AI generator
            data_dir = Path("data")
            ai_generator = AIInsightsGenerator(data_dir)
            
            if ai_generator.enabled:
                with st.spinner("🧠 Generating comprehensive ESG analysis..."):
                    # Prepare comprehensive context for AI
                    
                    # Get company ESG scores
                    esg_score_str = f"{esg_score:.2f}" if esg_score is not None else "N/A"
                    env_score_str = f"{env_score:.2f}" if env_score is not None else "N/A"
                    social_score_str = f"{social_score:.2f}" if social_score is not None else "N/A"
                    gov_score_str = f"{gov_score:.2f}" if gov_score is not None else "N/A"
                    
                    company_esg_summary = f"""
**{ticker} ESG Scores:**
- Total ESG Score: {esg_score_str}
- Environment Score: {env_score_str}
- Social Score: {social_score_str}
- Governance Score: {gov_score_str}
"""
                    
                    # Get competitor data if available
                    competitor_context = ""
                    try:
                        from utils.data_fetcher import get_competitor_esg_data
                        competitors = {
                            'QCOM': 'Qualcomm',
                            'AVGO': 'Broadcom',
                            'MRVL': 'Marvell Technology',
                            'AMD': 'AMD',
                            'INTC': 'Intel',
                            'TXN': 'Texas Instruments'
                        }
                        esg_comparison = get_competitor_esg_data(competitors)
                        if esg_comparison:
                            esg_comp_df = pd.DataFrame(esg_comparison)
                            competitor_context = f"\n**Industry Peer Comparison:**\n{esg_comp_df.to_string(index=False)}"
                    except:
                        competitor_context = "\n(Competitor data not available for this analysis)"
                    
                    # Get company context
                    company_context = f"""
**Company Information:**
- Sector: {info.get('sector', 'N/A')}
- Industry: {info.get('industry', 'N/A')}
- Employees: {info.get('fullTimeEmployees', 'N/A'):,}"""
                    
                    # Create comprehensive prompt
                    prompt = f"""You are a senior ESG (Environmental, Social, Governance) analyst providing a comprehensive analysis of {ticker}'s sustainability performance and practices.

{company_esg_summary}

{company_context}

{competitor_context}

**Visual Context:**
The analysis includes several key visualizations:
1. ESG Score Breakdown Chart: Shows Environment, Social, and Governance scores individually
2. Industry ESG Comparison Chart: Displays Total ESG Scores across semiconductor peers
3. ESG Component Scores Chart: Shows industry average for each ESG pillar

**Scoring System:**
ESG scores are RISK scores where LOWER is BETTER:
- 0-10: Negligible Risk
- 10-20: Low Risk  
- 20-30: Medium Risk
- 30-40: High Risk
- 40+: Severe Risk

**Your Task:**
Provide a comprehensive, insightful ESG analysis (500-700 words) that:

1. **Overall ESG Risk Profile**: Analyze {ticker}'s total ESG score. What does it reveal about the company's sustainability risk exposure and management quality?

2. **Component Analysis**: Deep dive into the three pillars shown in the breakdown chart:
   - Environment: What does the score suggest about climate risk, resource management, and environmental practices?
   - Social: Evaluate workforce, community, product safety, and social impact practices
   - Governance: Assess board structure, ethics, transparency, and corporate governance quality

3. **Competitive Positioning**: Reference the industry comparison chart. How does {ticker} stack up against Broadcom, AMD, Marvell, Intel, and Texas Instruments? Who are the ESG leaders and laggards in this peer group?

4. **Industry Context**: What does the semiconductor industry ESG landscape look like? Are there common challenges or opportunities?

5. **Risk Assessment**: Identify the most significant ESG risks for {ticker} based on the data. Which pillar needs the most attention?

6. **Strengths & Opportunities**: What are {ticker}'s ESG strengths? Where are the opportunities for improvement?

7. **Investment Implications**: How should investors interpret these ESG metrics? What do they mean for long-term sustainability and risk management?

8. **Forward-Looking Perspective**: What ESG trends should stakeholders monitor for {ticker} and the semiconductor industry?

Be specific, cite actual scores from the data, reference the visualizations and peer comparisons, and provide actionable insights. Remember that LOWER scores indicate BETTER ESG risk management. Write in a professional yet accessible tone suitable for institutional investors, ESG-focused funds, and sustainability-conscious stakeholders."""
                    
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
                                    "content": "You are a senior ESG analyst with deep expertise in sustainability, corporate governance, and responsible investing. Provide comprehensive, data-driven insights that reference specific metrics and visualizations. Remember that ESG scores are risk scores where lower is better."
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
                        st.markdown("""<div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50;"></div>""", unsafe_allow_html=True)
                        st.markdown(ai_insight)
                        
                        # Add disclaimer
                        st.caption("💡 AI-generated ESG analysis based on current sustainability data. This analysis references all ESG scores, charts, and peer comparisons shown above. ESG investing involves additional considerations beyond financial metrics.")
                        
                    except Exception as e:
                        st.warning(f"⚠️ Unable to generate AI ESG analysis: {str(e)}")
                        st.info("Please ensure your OpenAI API key is properly configured in the .env file.")
            else:
                st.info("🔑 **AI ESG Analysis Unavailable**: Configure your OpenAI API key in the .env file to enable comprehensive AI-powered ESG insights that analyze all the sustainability data, scores, and peer comparisons on this page.")
        
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
