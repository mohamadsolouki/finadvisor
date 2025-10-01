"""
ESG Analysis Page Module
Displays Environmental, Social, and Governance metrics
"""

import streamlit as st
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
