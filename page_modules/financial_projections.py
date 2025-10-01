"""
Financial Projections Page Module
Projects future financial statements and compares with company guidance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_fetcher import get_stock_info
import yfinance as yf


def calculate_historical_growth_rates(ticker):
    """Calculate historical growth rates from financial statements"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get financial statements
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        
        if income_stmt.empty:
            return None, None, None, None
        
        # Transpose to have years as rows
        income_stmt = income_stmt.T.sort_index()
        balance_sheet = balance_sheet.T.sort_index() if not balance_sheet.empty else pd.DataFrame()
        cashflow = cashflow.T.sort_index() if not cashflow.empty else pd.DataFrame()
        
        # Calculate growth rates
        growth_rates = {}
        
        # Revenue growth
        if 'Total Revenue' in income_stmt.columns:
            revenues = income_stmt['Total Revenue']
            revenue_growth = revenues.pct_change().mean()
            growth_rates['revenue'] = revenue_growth
        
        # Operating Income growth
        if 'Operating Income' in income_stmt.columns:
            op_income = income_stmt['Operating Income']
            op_income_growth = op_income.pct_change().mean()
            growth_rates['operating_income'] = op_income_growth
        
        # Net Income growth
        if 'Net Income' in income_stmt.columns:
            net_income = income_stmt['Net Income']
            net_income_growth = net_income.pct_change().mean()
            growth_rates['net_income'] = net_income_growth
        
        # EPS growth (if available)
        if 'Basic EPS' in income_stmt.columns:
            eps = income_stmt['Basic EPS']
            eps_growth = eps.pct_change().mean()
            growth_rates['eps'] = eps_growth
        
        # Free Cash Flow growth
        if 'Free Cash Flow' in cashflow.columns:
            fcf = cashflow['Free Cash Flow']
            fcf_growth = fcf.pct_change().mean()
            growth_rates['free_cash_flow'] = fcf_growth
        
        return income_stmt, balance_sheet, cashflow, growth_rates
        
    except Exception as e:
        st.error(f"Error fetching financial statements: {str(e)}")
        return None, None, None, None


def project_financials(base_values, growth_rates, years=5):
    """Project financial statements into the future"""
    projections = {}
    
    for metric, base_value in base_values.items():
        if metric in growth_rates and not pd.isna(growth_rates[metric]):
            growth_rate = growth_rates[metric]
            projection = [base_value]
            
            for year in range(1, years + 1):
                # Apply compound growth
                projected_value = base_value * ((1 + growth_rate) ** year)
                projection.append(projected_value)
            
            projections[metric] = projection
    
    return projections


def get_analyst_estimates(ticker):
    """Get analyst estimates and company guidance if available"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get analyst recommendations and targets
        recommendations = stock.recommendations
        analyst_price_target = stock.info.get('targetMeanPrice', None)
        
        # Get forward estimates
        earnings_estimates = {}
        if hasattr(stock, 'calendar') and stock.calendar is not None:
            earnings_estimates = stock.calendar
        
        # Get growth estimates
        growth_estimates = None
        if hasattr(stock, 'earnings_forecasts'):
            growth_estimates = stock.earnings_forecasts
        
        return {
            'price_target': analyst_price_target,
            'recommendations': recommendations,
            'earnings_estimates': earnings_estimates,
            'growth_estimates': growth_estimates
        }
    except Exception as e:
        return None


def display_financial_projections(ticker, cached_info):
    """Display financial projections page"""
    st.subheader("🔮 Financial Projections & Forecasting")
    st.markdown(f"Project future financial performance for **{ticker}** and compare with analyst expectations")
    
    # Handle None cached_info
    if cached_info is None:
        cached_info = {}
    
    # Projection settings
    col1, col2 = st.columns([2, 1])
    
    with col1:
        projection_years = st.slider(
            "Projection Period (Years)",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of years to project into the future"
        )
    
    with col2:
        st.markdown("### Projection Method")
        st.info("📊 **Historical Growth Rate Method**\nBased on average historical growth")
    
    # Fetch historical data and calculate growth rates
    with st.spinner("📊 Analyzing historical financial statements..."):
        income_stmt, balance_sheet, cashflow, growth_rates = calculate_historical_growth_rates(ticker)
    
    if income_stmt is None or growth_rates is None or len(growth_rates) == 0:
        st.error("⚠️ Unable to fetch historical financial data for projections")
        st.info("💡 This feature requires access to detailed financial statements. Some companies may have limited data availability.")
        return
    
    # Display historical data summary
    st.markdown("---")
    st.markdown("### 📈 Historical Financial Performance")
    
    col1, col2, col3 = st.columns(3)
    
    # Get most recent values
    most_recent_date = income_stmt.index[-1]
    
    with col1:
        if 'Total Revenue' in income_stmt.columns:
            latest_revenue = income_stmt['Total Revenue'].iloc[-1]
            revenue_growth = growth_rates.get('revenue', 0) * 100
            st.metric(
                "Latest Revenue",
                f"${latest_revenue/1e9:.2f}B",
                f"{revenue_growth:.1f}% avg growth"
            )
    
    with col2:
        if 'Net Income' in income_stmt.columns:
            latest_net_income = income_stmt['Net Income'].iloc[-1]
            ni_growth = growth_rates.get('net_income', 0) * 100
            st.metric(
                "Latest Net Income",
                f"${latest_net_income/1e9:.2f}B",
                f"{ni_growth:.1f}% avg growth"
            )
    
    with col3:
        if 'Operating Income' in income_stmt.columns:
            latest_op_income = income_stmt['Operating Income'].iloc[-1]
            op_growth = growth_rates.get('operating_income', 0) * 100
            st.metric(
                "Latest Operating Income",
                f"${latest_op_income/1e9:.2f}B",
                f"{op_growth:.1f}% avg growth"
            )
    
    st.caption(f"📅 Data as of: {most_recent_date.strftime('%Y-%m-%d')}")
    
    # Display historical growth rates
    st.markdown("---")
    st.markdown("### 📊 Historical Growth Rates")
    st.markdown("Average annual growth rates calculated from historical financial statements")
    
    growth_df = pd.DataFrame([
        {
            'Metric': 'Revenue',
            'Annual Growth Rate': f"{growth_rates.get('revenue', 0)*100:.2f}%",
            'Status': '📈' if growth_rates.get('revenue', 0) > 0 else '📉'
        },
        {
            'Metric': 'Operating Income',
            'Annual Growth Rate': f"{growth_rates.get('operating_income', 0)*100:.2f}%",
            'Status': '📈' if growth_rates.get('operating_income', 0) > 0 else '📉'
        },
        {
            'Metric': 'Net Income',
            'Annual Growth Rate': f"{growth_rates.get('net_income', 0)*100:.2f}%",
            'Status': '📈' if growth_rates.get('net_income', 0) > 0 else '📉'
        },
        {
            'Metric': 'EPS',
            'Annual Growth Rate': f"{growth_rates.get('eps', 0)*100:.2f}%",
            'Status': '📈' if growth_rates.get('eps', 0) > 0 else '📉'
        } if 'eps' in growth_rates else None,
        {
            'Metric': 'Free Cash Flow',
            'Annual Growth Rate': f"{growth_rates.get('free_cash_flow', 0)*100:.2f}%",
            'Status': '📈' if growth_rates.get('free_cash_flow', 0) > 0 else '📉'
        } if 'free_cash_flow' in growth_rates else None
    ])
    
    growth_df = growth_df.dropna()
    st.dataframe(growth_df, use_container_width=True, hide_index=True)
    
    # Generate projections
    st.markdown("---")
    st.markdown("### 🔮 Financial Projections")
    
    base_values = {}
    if 'Total Revenue' in income_stmt.columns:
        base_values['revenue'] = income_stmt['Total Revenue'].iloc[-1]
    if 'Operating Income' in income_stmt.columns:
        base_values['operating_income'] = income_stmt['Operating Income'].iloc[-1]
    if 'Net Income' in income_stmt.columns:
        base_values['net_income'] = income_stmt['Net Income'].iloc[-1]
    if 'Basic EPS' in income_stmt.columns:
        base_values['eps'] = income_stmt['Basic EPS'].iloc[-1]
    
    projections = project_financials(base_values, growth_rates, projection_years)
    
    # Create projection years
    base_year = most_recent_date.year
    projection_years_list = [base_year] + [base_year + i for i in range(1, projection_years + 1)]
    
    # Create comprehensive projection chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Revenue Projection',
            'Net Income Projection',
            'Operating Income Projection',
            'EPS Projection'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # Revenue projection
    if 'revenue' in projections:
        revenue_proj = [v/1e9 for v in projections['revenue']]
        fig.add_trace(
            go.Scatter(
                x=projection_years_list[:len(income_stmt)],
                y=[income_stmt['Total Revenue'].iloc[i]/1e9 for i in range(len(income_stmt))],
                name='Historical Revenue',
                line=dict(color='#1f77b4', width=3),
                mode='lines+markers'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=projection_years_list,
                y=revenue_proj,
                name='Projected Revenue',
                line=dict(color='#ff7f0e', width=3, dash='dash'),
                mode='lines+markers'
            ),
            row=1, col=1
        )
    
    # Net Income projection
    if 'net_income' in projections:
        ni_proj = [v/1e9 for v in projections['net_income']]
        fig.add_trace(
            go.Scatter(
                x=projection_years_list[:len(income_stmt)],
                y=[income_stmt['Net Income'].iloc[i]/1e9 for i in range(len(income_stmt))],
                name='Historical Net Income',
                line=dict(color='#2ca02c', width=3),
                mode='lines+markers',
                showlegend=False
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=projection_years_list,
                y=ni_proj,
                name='Projected Net Income',
                line=dict(color='#d62728', width=3, dash='dash'),
                mode='lines+markers',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Operating Income projection
    if 'operating_income' in projections:
        oi_proj = [v/1e9 for v in projections['operating_income']]
        fig.add_trace(
            go.Scatter(
                x=projection_years_list[:len(income_stmt)],
                y=[income_stmt['Operating Income'].iloc[i]/1e9 for i in range(len(income_stmt))],
                name='Historical Op Income',
                line=dict(color='#9467bd', width=3),
                mode='lines+markers',
                showlegend=False
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=projection_years_list,
                y=oi_proj,
                name='Projected Op Income',
                line=dict(color='#8c564b', width=3, dash='dash'),
                mode='lines+markers',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # EPS projection
    if 'eps' in projections:
        eps_proj = projections['eps']
        historical_eps = income_stmt['Basic EPS'].values if 'Basic EPS' in income_stmt.columns else []
        fig.add_trace(
            go.Scatter(
                x=projection_years_list[:len(historical_eps)],
                y=historical_eps,
                name='Historical EPS',
                line=dict(color='#e377c2', width=3),
                mode='lines+markers',
                showlegend=False
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=projection_years_list,
                y=eps_proj,
                name='Projected EPS',
                line=dict(color='#7f7f7f', width=3, dash='dash'),
                mode='lines+markers',
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update axes
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    
    fig.update_yaxes(title_text="Revenue ($B)", row=1, col=1)
    fig.update_yaxes(title_text="Net Income ($B)", row=1, col=2)
    fig.update_yaxes(title_text="Operating Income ($B)", row=2, col=1)
    fig.update_yaxes(title_text="EPS ($)", row=2, col=2)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        template='plotly_white',
        title_text=f"{ticker} Financial Projections - {projection_years} Year Forecast",
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display projection table
    st.markdown("### 📋 Detailed Projections Table")
    
    projection_table = pd.DataFrame({
        'Year': projection_years_list,
        'Revenue ($B)': [v/1e9 for v in projections['revenue']] if 'revenue' in projections else [None] * len(projection_years_list),
        'Net Income ($B)': [v/1e9 for v in projections['net_income']] if 'net_income' in projections else [None] * len(projection_years_list),
        'Operating Income ($B)': [v/1e9 for v in projections['operating_income']] if 'operating_income' in projections else [None] * len(projection_years_list),
        'EPS ($)': projections.get('eps', [None] * len(projection_years_list))
    })
    
    st.dataframe(projection_table, use_container_width=True, hide_index=True)
    
    # Get analyst estimates
    st.markdown("---")
    st.markdown("### 🎯 Analyst Estimates & Company Guidance")
    
    analyst_data = get_analyst_estimates(ticker)
    
    if analyst_data and analyst_data.get('price_target'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Analyst Price Target",
                f"${analyst_data['price_target']:.2f}",
                help="Mean analyst price target"
            )
        
        with col2:
            current_price = cached_info.get('currentPrice', 0)
            if current_price > 0:
                upside = ((analyst_data['price_target'] - current_price) / current_price) * 100
                st.metric(
                    "Potential Upside",
                    f"{upside:.1f}%",
                    help="Upside to analyst target price"
                )
    else:
        st.info("📊 Analyst estimates and company guidance data limited for this ticker")
    
    # Get forward P/E and growth expectations
    forward_pe = cached_info.get('forwardPE', None)
    peg_ratio = cached_info.get('pegRatio', None)
    
    if forward_pe or peg_ratio:
        col1, col2 = st.columns(2)
        
        with col1:
            if forward_pe:
                st.metric("Forward P/E", f"{forward_pe:.2f}", help="Forward price-to-earnings ratio")
        
        with col2:
            if peg_ratio:
                st.metric("PEG Ratio", f"{peg_ratio:.2f}", help="Price/Earnings to Growth ratio")
    
    # AI Analysis Section
    st.markdown("---")
    st.markdown("### 🤖 AI-Powered Projection Analysis")
    st.markdown("Comprehensive interpretation of projections and comparison with market expectations")
    
    # Import AI insights generator
    from utils.ai_insights_generator import AIInsightsGenerator
    from pathlib import Path
    
    # Initialize AI generator
    data_dir = Path("data")
    ai_generator = AIInsightsGenerator(data_dir)
    
    if ai_generator.enabled:
        with st.spinner("🧠 Generating comprehensive projection analysis..."):
            # Prepare comprehensive context
            projection_summary = f"""
**Financial Projection Analysis for {ticker}:**

**Base Year ({base_year}) Financials:**
- Revenue: ${base_values.get('revenue', 0)/1e9:.2f}B
- Net Income: ${base_values.get('net_income', 0)/1e9:.2f}B
- Operating Income: ${base_values.get('operating_income', 0)/1e9:.2f}B
- EPS: ${base_values.get('eps', 0):.2f}

**Historical Growth Rates (Used for Projection):**
- Revenue Growth: {growth_rates.get('revenue', 0)*100:.2f}% annually
- Net Income Growth: {growth_rates.get('net_income', 0)*100:.2f}% annually
- Operating Income Growth: {growth_rates.get('operating_income', 0)*100:.2f}% annually
- EPS Growth: {growth_rates.get('eps', 0)*100:.2f}% annually

**{projection_years}-Year Projections:**
"""
            
            # Add final year projections
            if 'revenue' in projections:
                final_revenue = projections['revenue'][-1]/1e9
                projection_summary += f"- Projected Revenue ({base_year + projection_years}): ${final_revenue:.2f}B\n"
            
            if 'net_income' in projections:
                final_ni = projections['net_income'][-1]/1e9
                projection_summary += f"- Projected Net Income ({base_year + projection_years}): ${final_ni:.2f}B\n"
            
            if 'eps' in projections:
                final_eps = projections['eps'][-1]
                projection_summary += f"- Projected EPS ({base_year + projection_years}): ${final_eps:.2f}\n"
            
            # Pre-format market context values to avoid f-string issues
            current_price = cached_info.get('currentPrice', 0)
            current_price_str = f"${current_price:.2f}" if current_price else "N/A"
            
            forward_pe_str = f"{forward_pe:.2f}" if forward_pe else "N/A"
            peg_ratio_str = f"{peg_ratio:.2f}" if peg_ratio else "N/A"
            
            analyst_target = analyst_data.get('price_target') if analyst_data else None
            analyst_target_str = f"${analyst_target:.2f}" if analyst_target else "N/A"
            
            market_cap = cached_info.get('marketCap', 0)
            market_cap_str = f"${market_cap/1e9:.2f}B" if market_cap else "N/A"
            
            # Add market context
            market_context = f"""
**Market Expectations & Valuation:**
- Current Stock Price: {current_price_str}
- Forward P/E: {forward_pe_str}
- PEG Ratio: {peg_ratio_str}
- Analyst Price Target: {analyst_target_str}
- Market Cap: {market_cap_str}
"""
            
            # Create comprehensive prompt
            prompt = f"""You are a senior financial analyst and equity research professional specializing in financial modeling and forecasting.

{projection_summary}

{market_context}

**Projection Methodology:**
The projections use a Historical Growth Rate method, calculating the average annual growth rate from historical financial statements and applying compound growth forward. This is one of several accepted projection methods (others include DCF, comparable company analysis, management guidance extrapolation).

**Visual Context:**
The projection dashboard shows 4 key charts:
1. **Revenue Projection**: Historical revenue trend (solid line) vs projected growth (dashed line)
2. **Net Income Projection**: Historical profitability vs future projections
3. **Operating Income Projection**: Operating performance trajectory
4. **EPS Projection**: Earnings per share historical and projected path

**Your Task:**
Provide a comprehensive financial projection analysis (700-900 words) that covers:

1. **Projection Methodology Assessment**: Evaluate the historical growth rate method used. What are its strengths and limitations? Is {ticker}'s historical growth rate sustainable or likely to change?

2. **Revenue Projection Analysis**: Analyze the {growth_rates.get('revenue', 0)*100:.2f}% revenue growth assumption. Reference the revenue chart - is this growth rate conservative, aggressive, or reasonable? What factors could accelerate or decelerate revenue growth?

3. **Profitability Trajectory**: Examine the net income and operating income projections. Are margins expanding or contracting? Is the {growth_rates.get('net_income', 0)*100:.2f}% net income growth rate aligned with revenue growth? Reference both profitability charts.

4. **EPS Growth Analysis**: Evaluate the projected EPS path shown in the chart. The {growth_rates.get('eps', 0)*100:.2f}% EPS growth - is this achievable? What role do share buybacks, margin expansion, or other factors play?

5. **Comparison with Market Expectations**: Compare our projections with:
   - Analyst price target (implies certain growth expectations)
   - Forward P/E ratio (market's growth assumptions)
   - PEG ratio (growth vs valuation)
   
   Are our projections more optimistic or conservative than the market? Explain the differences.

6. **Key Assumptions & Risks**: What assumptions underpin these projections? What could cause actual results to differ significantly from projections (both upside and downside scenarios)?

7. **Industry & Competitive Context**: How do these growth rates compare to the semiconductor industry? Is {ticker} gaining or losing market share at these growth rates?

8. **Valuation Implications**: Based on the projected financials, does the current stock price and forward P/E seem reasonable? If the projections materialize, what might the stock be worth in {projection_years} years?

9. **Alternative Scenarios**: What would bull case and bear case projections look like? What different growth rates would be more/less aggressive?

10. **Investment Implications**: Based on the projection analysis and comparison with market expectations, what's the investment thesis? Are the projections achievable and attractive for investors?

Be specific, cite actual numbers from the projections and metrics, reference all four projection charts, compare with market expectations, and provide balanced analysis. Acknowledge both opportunities and risks. Write in a professional tone suitable for equity research reports and institutional investors."""
            
            try:
                # Generate AI insight using OpenAI
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
                            "content": "You are a senior financial analyst and equity research professional with deep expertise in financial modeling, forecasting, and valuation. Provide comprehensive, balanced analysis that compares internal projections with market expectations and explains differences professionally."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                
                ai_insight = response.choices[0].message.content.strip()
                
                # Display AI insight in an attractive format
                st.markdown(f"""
                <div style="background-color: #e8f4f8; padding: 25px; border-radius: 10px; 
                            border-left: 5px solid #00acc1; line-height: 1.8; 
                            color: #212529; white-space: pre-wrap;">
                {ai_insight}
                </div>
                """, unsafe_allow_html=True)
                
                # Add disclaimer
                st.caption("💡 AI-generated projection analysis based on historical financial data and market expectations. Projections are estimates based on historical growth rates and may not reflect future performance. These projections reference all charts and compare with analyst expectations shown above. Always conduct additional research and consider multiple scenarios before making investment decisions.")
                
            except Exception as e:
                st.warning(f"⚠️ Unable to generate AI projection analysis: {str(e)}")
                st.info("Please ensure your OpenAI API key is properly configured in the .env file.")
    else:
        st.info("🔑 **AI Projection Analysis Unavailable**: Configure your OpenAI API key in the .env file to enable comprehensive AI-powered analysis that compares our projections with company guidance and analyst expectations, and explains the differences in detail.")
