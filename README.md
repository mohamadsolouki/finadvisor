# Qualcomm Financial Analysis Dashboard 📊

A comprehensive Streamlit application for analyzing Qualcomm's financial data with interactive visualizations, real-time market data, and exportable reports.

## Features

✨ **Key Features:**
- 📈 Interactive financial data visualization
- 📊 Real-time stock price analysis with Yahoo Finance integration
- 🌱 ESG (Environmental, Social, Governance) analysis
- 🏭 Industry benchmarking with peer comparison
- ⚠️ Comprehensive risk analysis (VaR, volatility, drawdown)
- 🎯 Automated data categorization into 10+ financial sections
- 💡 AI-powered financial insights and analysis
- 📄 Comprehensive PDF report generation
- 📊 Detailed Excel export with multiple sheets
- 🎨 Professional and intuitive UI
- 🔍 Custom analysis tools for metric comparison

## Analysis Pages

### 1. **Executive Summary**
Quick overview with key metrics and performance trends

### 2. **Financial Analysis** (Consolidated View)
Complete financial statements and ratios in one page:
- Income Statement - Revenue, profit metrics, EPS
- Balance Sheet - Assets, liabilities, equity
- Cash Flow - Operating, investing, financing activities
- Growth Ratios - Sales, income, EPS growth trends
- Equity Ratios - EPS, dividends, book value
- Profitability Ratios - Margins, ROE, ROA
- Cost Ratios - Operating costs, administration
- Liquidity Ratios - Current ratio, quick ratio
- Leverage Ratios - Debt/equity, capital structure
- Efficiency Ratios - Turnover metrics, utilization

### 3. **Price Analysis**
Real-time stock market data:
- Current price and daily changes
- 52-week high/low tracking
- Price history with 50-day and 200-day moving averages
- Trading volume analysis
- Return distribution
- Performance metrics (1M, 3M, 1Y, YTD)

### 4. **ESG Analysis**
Environmental, Social, and Governance metrics:
- Total ESG score
- Individual component scores
- Detailed sustainability metrics
- Score interpretation guide

### 5. **Industry Benchmarking**
Compare with semiconductor industry peers:
- Market cap comparison
- P/E and P/B ratios
- Profitability metrics
- Peer comparison with NVIDIA, Broadcom, Texas Instruments, AMD
- Visual comparison charts

### 6. **Risk Analysis**
Comprehensive risk metrics:
- Beta (market volatility comparison)
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Value at Risk (VaR) at 95% and 99% confidence
- Rolling volatility charts
- Drawdown analysis

### 7. **Custom Analysis**
Create custom metric comparisons across all categories

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the repository**
   ```bash
   cd d:\git\finadvisor
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data file location**
   Ensure `QCOMfinancials.csv` is in the `data/` folder:
   ```
   finadvisor/
   ├── app.py
   ├── requirements.txt
   ├── data/
   │   └── QCOMfinancials.csv
   └── utils/
       ├── __init__.py
       ├── data_analyzer.py
       └── report_generator.py
   ```

## Usage

### Running the Application

1. **Start the Streamlit server**
   ```bash
   streamlit run app.py
   ```

2. **Access the dashboard**
   - The application will automatically open in your default browser
   - Default URL: `http://localhost:8501`

### Navigation

The application provides several views accessible from the sidebar:

#### 📋 Executive Summary
- Key financial metrics at a glance
- Year-over-year comparisons
- Revenue and profit trend visualizations
- AI-generated insights

#### 💰 Income Statement
- Revenue breakdown
- Profit metrics
- EPS analysis
- Trend visualizations

#### 🏦 Balance Sheet
- Asset composition
- Liability structure
- Equity analysis
- Financial position trends

#### 💵 Cash Flow
- Operating cash flow
- Investing activities
- Financing activities
- Cash flow trends

#### 📊 Financial Ratios
- Comprehensive ratio analysis
- Multi-year comparisons
- Performance benchmarking
- Trend identification

#### 🔍 All Categories
- Complete financial overview
- All sections in one view
- Comprehensive analysis

#### 🛠️ Custom Analysis
- Select specific metrics
- Create custom comparisons
- Build tailored visualizations

### Export Options

#### 📄 PDF Report
1. Click "Generate PDF Report" in the sidebar
2. Wait for processing (5-10 seconds)
3. Download the comprehensive PDF report
4. Report includes:
   - Executive summary
   - Financial health score
   - All data categories
   - Insights and analysis

#### 📊 Excel Report
1. Click "Generate Excel Report" in the sidebar
2. Wait for processing (3-5 seconds)
3. Download the Excel workbook
4. Report includes:
   - Summary sheet with insights
   - Separate sheet for each category
   - Full data sheet
   - Professional formatting

## Project Structure

```
finadvisor/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
│
├── data/
│   └── QCOMfinancials.csv     # Financial data (2020-2024)
│
└── utils/
    ├── __init__.py            # Package initializer
    ├── data_analyzer.py       # Financial analysis engine
    └── report_generator.py    # PDF/Excel report generation
```

## Technical Details

### Dependencies

- **streamlit** - Web application framework
- **pandas** - Data manipulation and analysis
- **plotly** - Interactive visualizations
- **reportlab** - PDF generation
- **openpyxl** - Excel file handling
- **numpy** - Numerical computations

### Data Format

The application expects CSV data with:
- First column: Parameter names
- Second column: Currency/unit
- Remaining columns: Years (2020-2024)
- Category headers in the Parameters column

### Analysis Features

**FinancialAnalyzer** provides:
- CAGR calculations
- Trend analysis
- Category-specific insights
- Financial health scoring
- Ratio calculations

**ReportGenerator** creates:
- Professional PDF reports with tables and insights
- Excel workbooks with formatted sheets
- Automated styling and formatting
- Chart-ready data organization

## Troubleshooting

### Common Issues

1. **Module not found errors**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Data file not found**
   - Verify `QCOMfinancials.csv` is in the `data/` folder
   - Check file name spelling (case-sensitive)

3. **Port already in use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

4. **PDF generation fails**
   - Ensure reportlab is installed correctly
   - Check system permissions for file writing

5. **Excel export issues**
   - Verify openpyxl installation
   - Check available memory for large datasets

## Customization

### Adding New Metrics

1. Update the CSV file with new parameters
2. The application will automatically detect and categorize
3. Insights can be customized in `utils/data_analyzer.py`

### Styling Changes

- Modify CSS in `app.py` (line 23-45)
- Update chart colors in chart creation functions
- Customize report styles in `utils/report_generator.py`

### New Visualizations

Add custom charts in `app.py`:
```python
def create_custom_chart(data, title):
    # Your custom Plotly chart logic
    return fig
```

## Performance Tips

- Application uses caching for data loading
- Reports are generated on-demand
- Large datasets may take longer to export
- Browser performance may vary with chart complexity

## Future Enhancements

Potential additions:
- [ ] Multi-company comparison
- [ ] Predictive analytics
- [ ] Industry benchmarking
- [ ] Real-time data updates
- [ ] Interactive forecasting
- [ ] More chart types

## Support

For issues, questions, or suggestions:
1. Check the troubleshooting section
2. Review the code comments
3. Verify data format matches expected structure

## License

This project is provided as-is for financial analysis purposes.

## Acknowledgments

- Built with Streamlit
- Data visualization powered by Plotly
- Financial data: Qualcomm Corporation (2020-2024)

---

**Created:** October 2025  
**Version:** 1.0.0  
**Python:** 3.8+
