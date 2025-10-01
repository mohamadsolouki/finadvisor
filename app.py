"""
Refactored Financial Advisor App - Main Entry Point
All display logic has been moved to separate page modules for better organization
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import time

# Import utilities
from utils.data_analyzer import FinancialAnalyzer
from utils.report_generator import ReportGenerator
from utils.data_fetcher import get_stock_info
from utils.visualizations import create_trend_chart, create_comparison_chart

# Import page modules
from page_modules.price_analysis import display_price_analysis
from page_modules.esg_analysis import display_esg_analysis

# Ticker symbol for Qualcomm
TICKER = "QCOM"

# Page configuration
st.set_page_config(
    page_title="Qualcomm Financial Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_categorize_data(file_path):
    """Load and categorize financial data into sections"""
    df = pd.read_csv(file_path)
    
    categories = {
        'Income Statement': [],
        'Balance Sheet': [],
        'Cash Flow': [],
        'Growth Ratios': [],
        'Equity Ratios': [],
        'Profitability Ratios': [],
        'Cost Ratios': [],
        'Liquidity Ratios': [],
        'Leverage Ratios': [],
        'Efficiency Ratios': []
    }
    
    current_category = None
    
    # Map CSV headers to category names
    header_to_category = {
        'Income Statements': 'Income Statement',
        'Balance Sheet': 'Balance Sheet',
        'Cash Flow': 'Cash Flow',
        'Key Ratios': None  # This is followed by subcategories
    }
    
    for idx, row in df.iterrows():
        param = str(row['Parameters']).strip()
        
        # Check if this is a main category header (Income Statements, Balance Sheet, etc.)
        if param in header_to_category:
            if header_to_category[param]:
                current_category = header_to_category[param]
            # For Key Ratios, wait for subcategory
            continue
        elif param in categories.keys():
            # This handles subcategories under Key Ratios
            current_category = param
        elif current_category and param != 'nan' and not pd.isna(row['Parameters']):
            categories[current_category].append(idx)
    
    # Create separate dataframes for each category
    categorized_data = {}
    for category, indices in categories.items():
        if indices:
            categorized_data[category] = df.loc[indices].reset_index(drop=True)
    
    return df, categorized_data


def display_category_data(category_name, data, analyzer):
    """Display data for a specific category with visualizations"""
    st.markdown(f'<div class="section-header">{category_name}</div>', unsafe_allow_html=True)
    
    # Display the data table
    st.dataframe(data, use_container_width=True, height=min(len(data) * 35 + 38, 400))
    
    # Create visualizations based on category
    col1, col2 = st.columns(2)
    
    with col1:
        if len(data) > 0:
            # Single metric trend
            metric_row = data.iloc[0:1]
            fig = create_trend_chart(metric_row, f"{data.iloc[0]['Parameters']} Trend", data.iloc[0]['Parameters'])
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if len(data) > 1:
            # Multiple metrics comparison
            comparison_data = data.iloc[:min(5, len(data))]
            fig = create_comparison_chart(comparison_data, f"{category_name} - Top Metrics Comparison")
            st.plotly_chart(fig, use_container_width=True)
    
    # Display insights
    insights = analyzer.get_category_insights(category_name, data)
    if insights:
        st.info(f"**Key Insights:** {insights}")
    
    st.markdown("---")


def main():
    # Header
    st.markdown('<div class="main-header">📊 Qualcomm Financial Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Initialize session state for data persistence
    if 'last_fetch_time' not in st.session_state:
        st.session_state.last_fetch_time = None
    
    # Sidebar Navigation
    st.sidebar.title("📊 Navigation")
    
    # Navigation pages
    page = st.sidebar.radio(
        "Pages",
        ["Executive Summary", "Financial Analysis", "Price Analysis", 
         "ESG Analysis", "Industry Benchmarking", "Risk Analysis", "Custom Analysis"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Load data
    data_path = Path(__file__).parent / "data" / "QCOMfinancials.csv"
    
    try:
        full_data, categorized_data = load_and_categorize_data(data_path)
        analyzer = FinancialAnalyzer(categorized_data)
        report_gen = ReportGenerator(full_data, categorized_data, analyzer)
        
        st.sidebar.markdown("---")
        
        # Data refresh section
        st.sidebar.subheader("⚙️ Data Management")
        
        if st.sidebar.button("🔄 Clear Cache & Refresh Data"):
            st.cache_data.clear()
            st.session_state.last_fetch_time = None
            st.sidebar.success("✅ Cache cleared!")
            st.sidebar.warning("⏱️ Please wait 30 seconds before navigating to avoid rate limits.")
            time.sleep(1)
            st.rerun()
        
        # Show last fetch time
        if st.session_state.last_fetch_time:
            time_diff = datetime.now() - st.session_state.last_fetch_time
            minutes = time_diff.seconds // 60
            st.sidebar.caption(f"📅 Last refreshed: {minutes} min ago" if minutes > 0 else "📅 Just refreshed")
        
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **📊 Data Sources:**
        - Financial Statements: CSV file
        - Market Data: Yahoo Finance API
        
        **⚡ Performance:**
        - Data cached for 2 hours
        - Reduces API calls
        - Avoids rate limiting
        
        **💡 Tips:**
        - If you see errors, wait 5-10 min
        - Use refresh button sparingly
        - Data updates automatically
        """)
        
        # Fetch Yahoo Finance data once and cache it
        if page in ["Price Analysis", "ESG Analysis", "Industry Benchmarking", "Risk Analysis"]:
            with st.spinner("Loading market data..."):
                cached_info = get_stock_info(TICKER)
                if not st.session_state.last_fetch_time:
                    st.session_state.last_fetch_time = datetime.now()
        else:
            cached_info = None
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Export Options")
        
        # Export buttons
        if st.sidebar.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                pdf_buffer = report_gen.generate_pdf_report()
                st.sidebar.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_buffer,
                    file_name=f"Qualcomm_Financial_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        if st.sidebar.button("📊 Generate Excel Report", use_container_width=True):
            with st.spinner("Generating Excel report..."):
                excel_buffer = report_gen.generate_excel_report()
                st.sidebar.download_button(
                    label="⬇️ Download Excel",
                    data=excel_buffer,
                    file_name=f"Qualcomm_Financial_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        # Main content based on page selection
        if page == "Executive Summary":
            display_executive_summary(categorized_data, analyzer)
        
        elif page == "Financial Analysis":
            display_financial_analysis(categorized_data, analyzer)
        
        elif page == "Price Analysis":
            display_price_analysis(TICKER, cached_info)
        
        elif page == "ESG Analysis":
            display_esg_analysis(TICKER, cached_info)
        
        elif page == "Industry Benchmarking":
            # Import here to avoid circular imports
            from page_modules.industry_benchmarking import display_industry_benchmarking
            display_industry_benchmarking(TICKER, cached_info)
        
        elif page == "Risk Analysis":
            # Import here to avoid circular imports
            from page_modules.risk_analysis import display_risk_analysis
            display_risk_analysis(TICKER, cached_info)
        
        elif page == "Custom Analysis":
            display_custom_analysis(categorized_data)
    
    except FileNotFoundError:
        st.error(f"❌ Data file not found at: {data_path}")
        st.info("Please make sure the file 'QCOMfinancials.csv' is in the 'data' folder.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)


def display_executive_summary(categorized_data, analyzer):
    """Display executive summary page"""
    st.subheader("Executive Summary")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Get latest year data - extract years from any available data
    latest_year = None
    prev_year = None
    
    # Try to get years from income data first, then fallback to other categories
    income_data = categorized_data.get('Income Statement')
    if income_data is not None and len(income_data) > 0:
        years = [col for col in income_data.columns if col not in ['Parameters', 'Currency']]
        if len(years) > 0:
            latest_year = years[-1]
            prev_year = years[-2] if len(years) > 1 else latest_year
    
    # If years not found in income data, try equity or profitability data
    if latest_year is None:
        for category in ['Equity Ratios', 'Profitability Ratios', 'Liquidity Ratios']:
            data = categorized_data.get(category)
            if data is not None and len(data) > 0:
                years = [col for col in data.columns if col not in ['Parameters', 'Currency']]
                if len(years) > 0:
                    latest_year = years[-1]
                    prev_year = years[-2] if len(years) > 1 else latest_year
                    break
    
    # Revenue metrics
    if income_data is not None and len(income_data) > 0 and latest_year is not None:
        # Revenue
        revenue_row = income_data[income_data['Parameters'] == 'Total Revenue']
        if not revenue_row.empty:
            current_revenue = float(str(revenue_row[latest_year].iloc[0]).replace(',', ''))
            prev_revenue = float(str(revenue_row[prev_year].iloc[0]).replace(',', ''))
            revenue_change = ((current_revenue - prev_revenue) / prev_revenue) * 100
            
            with col1:
                st.metric("Total Revenue (USD M)", f"${current_revenue:,.0f}", 
                         f"{revenue_change:+.2f}%")
        
        # Net Income
        net_income_row = income_data[income_data['Parameters'] == 'Net Income']
        if not net_income_row.empty:
            current_ni = float(str(net_income_row[latest_year].iloc[0]).replace(',', ''))
            prev_ni = float(str(net_income_row[prev_year].iloc[0]).replace(',', ''))
            ni_change = ((current_ni - prev_ni) / prev_ni) * 100
            
            with col2:
                st.metric("Net Income (USD M)", f"${current_ni:,.0f}", 
                         f"{ni_change:+.2f}%")
    
    # EPS
    equity_data = categorized_data.get('Equity Ratios')
    if equity_data is not None and len(equity_data) > 0 and latest_year is not None:
        eps_row = equity_data[equity_data['Parameters'] == 'EPS (Earnings per Share)']
        if not eps_row.empty:
            current_eps = float(str(eps_row[latest_year].iloc[0]).replace(',', ''))
            prev_eps = float(str(eps_row[prev_year].iloc[0]).replace(',', ''))
            eps_change = ((current_eps - prev_eps) / prev_eps) * 100
            
            with col3:
                st.metric("EPS (USD)", f"${current_eps:.2f}", 
                         f"{eps_change:+.2f}%")
    
    # ROE
    profit_data = categorized_data.get('Profitability Ratios')
    if profit_data is not None and len(profit_data) > 0 and latest_year is not None:
        roe_row = profit_data[profit_data['Parameters'] == 'Return on Equity']
        if not roe_row.empty:
            current_roe = float(str(roe_row[latest_year].iloc[0]).replace(',', ''))
            
            with col4:
                st.metric("Return on Equity", f"{current_roe:.2f}%")
    
    st.markdown("---")
    
    # Revenue and Profit Trends
    st.subheader("Revenue and Profit Trends (2020-2024)")
    if income_data is not None:
        revenue_profit_data = income_data[income_data['Parameters'].isin(['Total Revenue', 'Net Income', 'Operating Income'])]
        fig = create_comparison_chart(revenue_profit_data, "Revenue and Profit Trends")
        st.plotly_chart(fig, use_container_width=True)
    
    # Key Insights
    st.subheader("Key Financial Insights")
    insights = analyzer.generate_executive_summary()
    for insight in insights:
        st.info(f"• {insight}")


def display_financial_analysis(categorized_data, analyzer):
    """Display complete financial analysis page with tabs"""
    st.subheader("📊 Complete Financial Analysis")
    st.markdown("Comprehensive view of all financial statements and key ratios")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📈 Financial Statements", "📊 Performance Ratios", "💹 Operational Ratios"])
    
    with tab1:
        st.markdown("### Financial Statements Overview")
        financial_statement_categories = ['Income Statement', 'Balance Sheet', 'Cash Flow']
        
        for category_name in financial_statement_categories:
            if category_name in categorized_data:
                display_category_data(category_name, categorized_data[category_name], analyzer)
    
    with tab2:
        st.markdown("### Performance and Growth Metrics")
        performance_categories = ['Growth Ratios', 'Equity Ratios', 'Profitability Ratios']
        
        for category_name in performance_categories:
            if category_name in categorized_data:
                display_category_data(category_name, categorized_data[category_name], analyzer)
    
    with tab3:
        st.markdown("### Operational and Financial Health Ratios")
        operational_categories = ['Cost Ratios', 'Liquidity Ratios', 'Leverage Ratios', 'Efficiency Ratios']
        
        for category_name in operational_categories:
            if category_name in categorized_data:
                display_category_data(category_name, categorized_data[category_name], analyzer)


def display_custom_analysis(categorized_data):
    """Display custom analysis page"""
    st.subheader("Custom Analysis")
    
    st.write("Select metrics to compare:")
    
    # Get all available metrics
    all_metrics = []
    for category_name, data in categorized_data.items():
        for param in data['Parameters'].tolist():
            all_metrics.append(f"{category_name}: {param}")
    
    selected_metrics = st.multiselect(
        "Choose metrics to visualize:",
        all_metrics,
        default=all_metrics[:3] if len(all_metrics) >= 3 else all_metrics
    )
    
    if selected_metrics:
        # Create custom comparison
        custom_data_rows = []
        
        for metric in selected_metrics:
            category, param = metric.split(': ', 1)
            data = categorized_data[category]
            row = data[data['Parameters'] == param]
            if not row.empty:
                custom_data_rows.append(row)
        
        if custom_data_rows:
            custom_df = pd.concat(custom_data_rows, ignore_index=True)
            
            st.dataframe(custom_df, use_container_width=True)
            
            fig = create_comparison_chart(custom_df, "Custom Metrics Comparison")
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
