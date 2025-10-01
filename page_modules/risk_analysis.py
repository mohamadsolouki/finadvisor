"""
Risk Analysis Page Module
Displays volatility, VaR, CVaR, and other risk metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    
    st.markdown("---")
    
    # All-in-One Comprehensive Risk Dashboard
    st.markdown("### 📊 Comprehensive Risk Dashboard")
    st.markdown("Unified view of all risk metrics and trends")
    
    # Create comprehensive subplot with 4 panels
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Price Performance & Cumulative Returns',
            'Rolling 30-Day Volatility (%)',
            'Drawdown Analysis (%)',
            'Return Distribution & VaR'
        ),
        specs=[
            [{"secondary_y": True}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # Panel 1: Price and Cumulative Returns
    fig.add_trace(
        go.Scatter(x=hist.index, y=hist['Close'], name='Price', 
                  line=dict(color='#1f77b4', width=2)),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=hist.index, y=hist['Cumulative']*100, name='Cumulative Return (%)',
                  line=dict(color='#2ca02c', width=2, dash='dot')),
        row=1, col=1, secondary_y=True
    )
    
    # Panel 2: Rolling Volatility
    fig.add_trace(
        go.Scatter(x=hist.index, y=hist['Rolling_Vol'], name='Rolling Volatility',
                  line=dict(color='#ff7f0e', width=2),
                  fill='tozeroy', fillcolor='rgba(255,127,14,0.2)'),
        row=1, col=2
    )
    # Add average volatility line
    avg_vol = hist['Rolling_Vol'].mean()
    fig.add_hline(y=avg_vol, line_dash="dash", line_color="red", 
                  annotation_text=f"Avg: {avg_vol:.1f}%",
                  row=1, col=2)
    
    # Panel 3: Drawdown
    fig.add_trace(
        go.Scatter(x=hist.index, y=hist['Drawdown'], name='Drawdown',
                  line=dict(color='#d62728', width=2),
                  fill='tozeroy', fillcolor='rgba(214,39,40,0.2)'),
        row=2, col=1
    )
    # Mark maximum drawdown
    max_dd_idx = hist['Drawdown'].idxmin()
    fig.add_trace(
        go.Scatter(x=[max_dd_idx], y=[max_drawdown], 
                  mode='markers', name='Max Drawdown',
                  marker=dict(size=12, color='red', symbol='x'),
                  showlegend=False),
        row=2, col=1
    )
    
    # Panel 4: Return Distribution with VaR
    hist_counts, bin_edges = np.histogram(hist['Returns']*100, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    fig.add_trace(
        go.Bar(x=bin_centers, y=hist_counts, name='Return Distribution',
              marker=dict(color='#9467bd', opacity=0.7)),
        row=2, col=2
    )
    # Add VaR line
    fig.add_vline(x=var_95*100, line_dash="dash", line_color="red",
                  annotation_text=f"VaR 95%: {var_95*100:.1f}%",
                  row=2, col=2)
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Daily Return (%)", row=2, col=2)
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Volatility (%)", row=1, col=2)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        template='plotly_white',
        title_text=f"{ticker} Comprehensive Risk Analysis Dashboard - {period} Period",
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("📊 **Dashboard Guide**: Top-left shows price and cumulative returns; Top-right displays volatility trends; Bottom-left shows drawdown periods; Bottom-right illustrates return distribution with VaR threshold")
    
    st.markdown("---")
    
    # AI-Powered Risk Analysis
    st.markdown("### 🤖 AI-Powered Risk Analysis")
    st.markdown("Comprehensive interpretation of risk metrics and portfolio implications")
    
    # Import AI insights generator
    from utils.ai_insights_generator import AIInsightsGenerator
    from pathlib import Path
    
    # Initialize AI generator
    data_dir = Path("data")
    ai_generator = AIInsightsGenerator(data_dir)
    
    if ai_generator.enabled:
        with st.spinner("🧠 Generating comprehensive risk analysis..."):
            # Calculate additional metrics for context
            positive_days = (hist['Returns'] > 0).sum()
            total_days = len(hist['Returns'])
            win_rate = (positive_days / total_days) * 100
            
            avg_positive_return = hist[hist['Returns'] > 0]['Returns'].mean() * 100
            avg_negative_return = hist[hist['Returns'] < 0]['Returns'].mean() * 100
            
            # Calculate downside deviation
            downside_returns = hist[hist['Returns'] < 0]['Returns']
            downside_deviation = downside_returns.std() * np.sqrt(252) * 100
            
            # Prepare comprehensive risk data
            risk_summary = f"""
**Risk Metrics Summary for {ticker} ({period} period):**

**Volatility Metrics:**
- Daily Volatility: {daily_vol*100:.2f}%
- Annualized Volatility: {annual_vol*100:.2f}%
- Average Rolling Volatility (30-day): {avg_vol:.2f}%
- Downside Deviation (annualized): {downside_deviation:.2f}%

**Value at Risk:**
- VaR (95% confidence): {var_95*100:.2f}%
- CVaR (95%): {cvar_95*100:.2f}%

**Drawdown Analysis:**
- Maximum Drawdown: {max_drawdown:.2f}%
- Maximum Drawdown Date: {max_dd_date.strftime('%Y-%m-%d')}

**Performance Metrics:**
- Annualized Return: {avg_return*100:.2f}%
- Sharpe Ratio: {sharpe_ratio:.2f}
- Win Rate: {win_rate:.1f}% (positive days)
- Average Positive Day: +{avg_positive_return:.2f}%
- Average Negative Day: {avg_negative_return:.2f}%
"""
            
            # Add Beta if available
            beta_context = ""
            try:
                if 'beta' in locals():
                    beta_context = f"\n**Market Sensitivity:**\n- Beta (vs S&P 500): {beta:.2f}"
            except:
                pass
            
            # Create comprehensive prompt
            prompt = f"""You are a senior risk analyst and portfolio manager providing a comprehensive risk analysis of {ticker}.

{risk_summary}
{beta_context}

**Visual Dashboard Context:**
The comprehensive risk dashboard shows 4 key visualizations:
1. **Price Performance & Cumulative Returns**: Shows both absolute price movement and total return over the period
2. **Rolling 30-Day Volatility**: Displays how volatility has changed over time, with average line
3. **Drawdown Analysis**: Illustrates peak-to-trough declines, with the maximum drawdown marked
4. **Return Distribution & VaR**: Shows the frequency distribution of daily returns with the VaR threshold

**Your Task:**
Provide a comprehensive, insightful risk analysis (600-800 words) that:

1. **Overall Risk Profile**: Characterize {ticker}'s risk level. Is it high, moderate, or low risk? How does the {annual_vol*100:.1f}% annualized volatility compare to typical stocks?

2. **Volatility Analysis**: Analyze the rolling volatility chart. Are there periods of elevated volatility? What might have caused volatility spikes? Is volatility trending up or down?

3. **Drawdown Assessment**: Interpret the {max_drawdown:.2f}% maximum drawdown. How severe is this? Reference the drawdown chart - were there multiple significant drawdowns or one major event? How long did recovery take?

4. **Value at Risk Interpretation**: Explain what the VaR ({var_95*100:.2f}%) and CVaR ({cvar_95*100:.2f}%) mean for investors. On a bad day (worst 5%), how much could be lost? Reference the return distribution chart.

5. **Risk-Adjusted Performance**: Analyze the Sharpe Ratio ({sharpe_ratio:.2f}). Is the return adequate compensation for the risk? How does the {win_rate:.1f}% win rate factor in?

6. **Asymmetric Risk**: Compare average positive days (+{avg_positive_return:.2f}%) vs negative days ({avg_negative_return:.2f}%). Is risk symmetric or are losses larger than gains?

7. **Portfolio Implications**: What does this risk profile mean for different investor types (conservative vs aggressive)? What position sizing would be appropriate?

8. **Risk Management Recommendations**: Based on all the data and charts, what risk management strategies should investors consider (stop losses, position limits, diversification, etc.)?

9. **Market Conditions Impact**: How might different market environments affect this risk profile? What scenarios pose the greatest risk?

10. **Forward-Looking Risk**: Based on the trends shown in the charts, is risk increasing or decreasing? What should investors monitor?

Be specific, cite actual numbers from the metrics, reference all four panels of the dashboard, and provide actionable risk management guidance. Write in a professional yet accessible tone suitable for institutional portfolio managers and sophisticated retail investors."""
            
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
                            "content": "You are a senior risk analyst and portfolio manager with deep expertise in quantitative risk analysis, volatility modeling, and portfolio risk management. Provide comprehensive, data-driven insights that reference specific metrics and all visualizations in the dashboard."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1800
                )
                
                ai_insight = response.choices[0].message.content.strip()
                
                # Display AI insight in an attractive format
                st.markdown("""<div style="background-color: #fff3cd; padding: 15px; border-radius: 10px; border-left: 5px solid #ffc107;"></div>""", unsafe_allow_html=True)
                st.markdown(ai_insight)
                
                # Add disclaimer
                st.caption("💡 AI-generated risk analysis based on historical data and statistical metrics. This analysis references all charts and risk metrics shown above. Past volatility and drawdowns do not guarantee future risk levels. Always conduct additional research and consult with financial advisors for portfolio decisions.")
                
            except Exception as e:
                st.warning(f"⚠️ Unable to generate AI risk analysis: {str(e)}")
                st.info("Please ensure your OpenAI API key is properly configured in the .env file.")
    else:
        st.info("🔑 **AI Risk Analysis Unavailable**: Configure your OpenAI API key in the .env file to enable comprehensive AI-powered risk insights that analyze all the metrics, trends, and the comprehensive dashboard on this page.")
