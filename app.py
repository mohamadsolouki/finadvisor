import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# Add utils directory to path
sys.path.append(str(Path(__file__).parent))

from utils.data_analyzer import FinancialAnalyzer
from utils.report_generator import ReportGenerator

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
    
    for idx, row in df.iterrows():
        param = str(row['Parameters']).strip()
        
        # Check if this is a category header
        if param in ['Income Statements', 'Balance Sheet', 'Cash Flow', 'Key Ratios']:
            continue
        elif param in categories.keys():
            current_category = param
        elif current_category and param != 'nan' and not pd.isna(row['Parameters']):
            categories[current_category].append(idx)
    
    # Create separate dataframes for each category
    categorized_data = {}
    for category, indices in categories.items():
        if indices:
            categorized_data[category] = df.loc[indices].reset_index(drop=True)
    
    return df, categorized_data


def create_trend_chart(data, title, metric_name):
    """Create an interactive trend chart"""
    years = [col for col in data.columns if col not in ['Parameters', 'Currency']]
    values = []
    
    for year in years:
        val = str(data[year].iloc[0]).replace(',', '')
        try:
            values.append(float(val))
        except:
            values.append(0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years,
        y=values,
        mode='lines+markers',
        name=metric_name,
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Value',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_comparison_chart(data, title):
    """Create a comparison chart for multiple metrics"""
    years = [col for col in data.columns if col not in ['Parameters', 'Currency']]
    
    fig = go.Figure()
    
    for idx, row in data.iterrows():
        values = []
        for year in years:
            val = str(row[year]).replace(',', '')
            try:
                values.append(float(val))
            except:
                values.append(0)
        
        fig.add_trace(go.Scatter(
            x=years,
            y=values,
            mode='lines+markers',
            name=row['Parameters']
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title='Value',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    return fig


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
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    # Load data
    data_path = Path(__file__).parent / "data" / "QCOMfinancials.csv"
    
    try:
        full_data, categorized_data = load_and_categorize_data(data_path)
        analyzer = FinancialAnalyzer(categorized_data)
        report_gen = ReportGenerator(full_data, categorized_data, analyzer)
        
        # Sidebar options
        page = st.sidebar.radio(
            "Select View",
            ["Executive Summary", "Income Statement", "Balance Sheet", "Cash Flow", 
             "Financial Ratios", "All Categories", "Custom Analysis"]
        )
        
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
            st.subheader("Executive Summary")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Get latest year data
            income_data = categorized_data.get('Income Statement')
            if income_data is not None and len(income_data) > 0:
                years = [col for col in income_data.columns if col not in ['Parameters', 'Currency']]
                latest_year = years[-1]
                prev_year = years[-2] if len(years) > 1 else latest_year
                
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
            if equity_data is not None and len(equity_data) > 0:
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
            if profit_data is not None and len(profit_data) > 0:
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
        
        elif page == "Income Statement":
            if 'Income Statement' in categorized_data:
                display_category_data('Income Statement', categorized_data['Income Statement'], analyzer)
        
        elif page == "Balance Sheet":
            if 'Balance Sheet' in categorized_data:
                display_category_data('Balance Sheet', categorized_data['Balance Sheet'], analyzer)
        
        elif page == "Cash Flow":
            if 'Cash Flow' in categorized_data:
                display_category_data('Cash Flow', categorized_data['Cash Flow'], analyzer)
        
        elif page == "Financial Ratios":
            st.subheader("Financial Ratios Analysis")
            
            ratio_categories = ['Growth Ratios', 'Equity Ratios', 'Profitability Ratios', 
                               'Cost Ratios', 'Liquidity Ratios', 'Leverage Ratios', 
                               'Efficiency Ratios']
            
            for category in ratio_categories:
                if category in categorized_data:
                    display_category_data(category, categorized_data[category], analyzer)
        
        elif page == "All Categories":
            st.subheader("Complete Financial Analysis")
            
            for category_name, data in categorized_data.items():
                display_category_data(category_name, data, analyzer)
        
        elif page == "Custom Analysis":
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
    
    except FileNotFoundError:
        st.error(f"❌ Data file not found at: {data_path}")
        st.info("Please make sure the file 'QCOMfinancials.csv' is in the 'data' folder.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
