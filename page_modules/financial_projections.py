"""
Financial Projections Page Module
Advanced financial projections with multiple methodologies, DCF valuation, and scenario analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_fetcher import get_stock_info
import yfinance as yf
from pathlib import Path
from scipy import stats
from datetime import datetime


def load_local_financial_data():
    """Load local financial data from CSV file (2020-2024)"""
    try:
        data_path = Path("data/QCOMfinancials.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)
            return df
        return None
    except Exception as e:
        st.warning(f"Could not load local financial data: {str(e)}")
        return None


def extract_historical_financials_from_csv(df):
    """Extract historical financial data from local CSV with enhanced metrics"""
    financials = {
        'years': [],
        'revenue': [],
        'net_income': [],
        'operating_income': [],
        'eps': [],
        'operating_cf': [],
        'free_cash_flow': [],
        'capex': [],
        'total_assets': [],
        'total_equity': [],
        'total_debt': [],
        'gross_profit': [],
        'cost_of_revenue': [],
        'current_assets': [],
        'current_liabilities': [],
        'interest_expense': []
    }
    
    try:
        # Get year columns
        year_cols = [col for col in df.columns if col.isdigit()]
        financials['years'] = sorted([int(y) for y in year_cols])
        
        # Extract metrics from CSV
        for _, row in df.iterrows():
            param = row['Parameters']
            
            if param == 'Total Revenue':
                financials['revenue'] = [float(str(row[str(y)]).replace(',', '')) for y in financials['years']]
            elif param == 'Net Income':
                financials['net_income'] = [float(str(row[str(y)]).replace(',', '')) for y in financials['years']]
            elif param == 'Operating Income':
                financials['operating_income'] = [float(str(row[str(y)]).replace(',', '')) for y in financials['years']]
            elif param == 'Diluted Normalized EPS':
                financials['eps'] = [float(str(row[str(y)]).replace(',', '')) for y in financials['years']]
            elif param == 'Cash from Operating Activities':
                financials['operating_cf'] = [float(str(row[str(y)]).replace(',', '')) for y in financials['years']]
            elif param == 'Capital Expenditure' or param == 'Capital Expenditures':
                financials['capex'] = [abs(float(str(row[str(y)]).replace(',', ''))) for y in financials['years']]
            elif param == 'Total Assets':
                financials['total_assets'] = [float(str(row[str(y)]).replace(',', '')) for y in financials['years']]
            elif param == 'Total Equity':
                financials['total_equity'] = [float(str(row[str(y)]).replace(',', '')) for y in financials['years']]
            elif param == 'Total Debt':
                financials['total_debt'] = [float(str(row[str(y)]).replace(',', '')) for y in financials['years']]
            elif param == 'Gross Profit':
                financials['gross_profit'] = [float(str(row[str(y)]).replace(',', '')) for y in financials['years']]
            elif param == 'Cost Of Revenue':
                financials['cost_of_revenue'] = [float(str(row[str(y)]).replace(',', '')) for y in financials['years']]
            elif param == 'Current Assets':
                financials['current_assets'] = [float(str(row[str(y)]).replace(',', '')) for y in financials['years']]
            elif param == 'Current Liabilities':
                financials['current_liabilities'] = [float(str(row[str(y)]).replace(',', '')) for y in financials['years']]
            elif param == 'Interest Expense':
                financials['interest_expense'] = [abs(float(str(row[str(y)]).replace(',', ''))) for y in financials['years']]
        
        # Calculate CapEx from investing cash flow if not available
        if not financials['capex'] and financials['revenue']:
            # Use industry average: Tech ~5%, Industrial ~8%, Retail ~3%
            # Default to 6% as reasonable estimate
            financials['capex'] = [rev * 0.06 for rev in financials['revenue']]
        
        # Calculate FCF = Operating CF - CapEx (proper formula)
        if financials['operating_cf'] and financials['capex']:
            financials['free_cash_flow'] = [
                cf - capex for cf, capex in zip(financials['operating_cf'], financials['capex'])
            ]
        elif financials['operating_cf']:
            # Fallback if no CapEx data
            financials['free_cash_flow'] = [
                cf * 0.75 for cf in financials['operating_cf']  # Assume CapEx is ~25% of OCF
            ]
        
        return financials
    except Exception as e:
        st.warning(f"Error extracting financial data: {str(e)}")
        return None


def calculate_historical_growth_rates(ticker, use_local_data=True):
    """Calculate comprehensive historical growth rates and metrics from financial statements"""
    try:
        # First try to load local data for complete historical view
        local_financials = None
        if use_local_data:
            df = load_local_financial_data()
            if df is not None:
                local_financials = extract_historical_financials_from_csv(df)
        
        # Also get yfinance data for additional metrics
        stock = yf.Ticker(ticker)
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        stock_info = stock.info if hasattr(stock, 'info') else {}
        
        # Combine local and yfinance data
        if local_financials and local_financials['years']:
            # Use local data as primary source (2020-2024)
            years = local_financials['years']
            historical_data = {
                'years': years,
                'revenue': local_financials['revenue'],
                'net_income': local_financials['net_income'],
                'operating_income': local_financials['operating_income'],
                'eps': local_financials['eps'],
                'operating_cf': local_financials['operating_cf'],
                'free_cash_flow': local_financials['free_cash_flow'],
                'capex': local_financials['capex'],
                'total_assets': local_financials['total_assets'],
                'total_equity': local_financials['total_equity'],
                'total_debt': local_financials['total_debt'],
                'gross_profit': local_financials['gross_profit'],
                'current_assets': local_financials.get('current_assets', []),
                'current_liabilities': local_financials.get('current_liabilities', []),
                'interest_expense': local_financials.get('interest_expense', [])
            }
        else:
            # Fallback to yfinance only
            if income_stmt.empty:
                return None, None
            
            income_stmt = income_stmt.T.sort_index()
            balance_sheet = balance_sheet.T.sort_index() if not balance_sheet.empty else pd.DataFrame()
            cashflow = cashflow.T.sort_index() if not cashflow.empty else pd.DataFrame()
            
            years = [d.year for d in income_stmt.index]
            historical_data = {
                'years': years,
                'revenue': income_stmt['Total Revenue'].values if 'Total Revenue' in income_stmt.columns else [],
                'net_income': income_stmt['Net Income'].values if 'Net Income' in income_stmt.columns else [],
                'operating_income': income_stmt['Operating Income'].values if 'Operating Income' in income_stmt.columns else [],
                'eps': income_stmt['Basic EPS'].values if 'Basic EPS' in income_stmt.columns else [],
                'operating_cf': cashflow['Operating Cash Flow'].values if 'Operating Cash Flow' in cashflow.columns else [],
                'free_cash_flow': cashflow['Free Cash Flow'].values if 'Free Cash Flow' in cashflow.columns else [],
                'total_assets': balance_sheet['Total Assets'].values if 'Total Assets' in balance_sheet.columns else [],
                'total_equity': balance_sheet['Total Equity Gross Minority Interest'].values if 'Total Equity Gross Minority Interest' in balance_sheet.columns else [],
                'gross_profit': income_stmt['Gross Profit'].values if 'Gross Profit' in income_stmt.columns else []
            }
        
        # Calculate comprehensive growth rates using multiple methods
        growth_rates = calculate_growth_metrics(historical_data)
        
        # Add industry context and validation
        industry_context = get_industry_context(ticker, stock_info)
        
        # Validate and adjust growth rates for reasonableness
        growth_rates = validate_growth_rates(growth_rates, historical_data, industry_context)
        
        return historical_data, growth_rates
        
    except Exception as e:
        st.error(f"Error fetching financial statements: {str(e)}")
        return None, None


def get_industry_context(ticker, stock_info):
    """Gather industry-specific context from yfinance for more reasonable assumptions"""
    context = {
        'industry': stock_info.get('industry', 'Unknown'),
        'sector': stock_info.get('sector', 'Unknown'),
        'market_cap': stock_info.get('marketCap', 0),
        'revenue_growth': stock_info.get('revenueGrowth', None),
        'earnings_growth': stock_info.get('earningsGrowth', None),
        'industry_pe': stock_info.get('forwardPE', None),
        'industry_peg': stock_info.get('pegRatio', None),
        'profit_margins': stock_info.get('profitMargins', None),
        'operating_margins': stock_info.get('operatingMargins', None),
        'roe': stock_info.get('returnOnEquity', None),
        'roa': stock_info.get('returnOnAssets', None),
        'debt_to_equity': stock_info.get('debtToEquity', None),
        'current_ratio': stock_info.get('currentRatio', None),
        'quick_ratio': stock_info.get('quickRatio', None),
        'beta': stock_info.get('beta', 1.0),
        'fifty_two_week_change': stock_info.get('52WeekChange', None)
    }
    
    # Determine company maturity based on market cap
    if context['market_cap'] > 200e9:  # > $200B
        context['maturity'] = 'mega_cap'
        context['typical_growth'] = 0.08  # 8%
    elif context['market_cap'] > 50e9:  # > $50B
        context['maturity'] = 'large_cap'
        context['typical_growth'] = 0.12  # 12%
    elif context['market_cap'] > 10e9:  # > $10B
        context['maturity'] = 'mid_cap'
        context['typical_growth'] = 0.15  # 15%
    else:
        context['maturity'] = 'small_cap'
        context['typical_growth'] = 0.20  # 20%
    
    # Industry-specific growth caps
    industry_growth_caps = {
        'Technology': 0.25,
        'Healthcare': 0.20,
        'Financial Services': 0.12,
        'Consumer Cyclical': 0.15,
        'Industrials': 0.10,
        'Energy': 0.15,
        'Utilities': 0.05,
        'Real Estate': 0.08,
        'Consumer Defensive': 0.08,
        'Basic Materials': 0.10,
        'Communication Services': 0.12
    }
    
    context['industry_growth_cap'] = industry_growth_caps.get(context['sector'], 0.15)
    
    return context


def validate_growth_rates(growth_rates, historical_data, industry_context):
    """Validate and adjust growth rates to be reasonable based on company size, industry, and historical trends"""
    validated_rates = growth_rates.copy()
    
    # Get industry-appropriate caps
    industry_cap = industry_context.get('industry_growth_cap', 0.15)
    maturity_cap = industry_context.get('typical_growth', 0.12)
    
    # Use the more conservative cap
    reasonable_cap = min(industry_cap, maturity_cap * 1.5)  # Allow some upside
    reasonable_floor = -0.10  # Don't project worse than -10% decline
    
    # Check for recent trend reversal or declining business
    if len(historical_data.get('revenue', [])) >= 2:
        recent_growth = (historical_data['revenue'][-1] - historical_data['revenue'][-2]) / historical_data['revenue'][-2]
        if recent_growth < 0:
            # Recent decline - be more conservative
            reasonable_cap = min(reasonable_cap, 0.05)  # Cap at 5% if declining
            st.warning("⚠️ Recent revenue decline detected - projections capped at 5% growth")
    
    # Validate each methodology
    for method in ['cagr', 'simple_average', 'weighted_average', 'linear_regression']:
        if method in validated_rates:
            for metric in validated_rates[method]:
                original_rate = validated_rates[method][metric]
                
                # Apply caps
                adjusted_rate = max(reasonable_floor, min(reasonable_cap, original_rate))
                
                # If we had to adjust significantly, note it
                if abs(adjusted_rate - original_rate) > 0.05:  # More than 5% difference
                    validated_rates[method][metric] = adjusted_rate
    
    return validated_rates


def calculate_growth_metrics(historical_data):
    """Calculate comprehensive growth metrics using multiple methodologies"""
    growth_rates = {
        'simple_average': {},
        'cagr': {},
        'linear_regression': {},
        'weighted_average': {},
        'volatility': {}
    }
    
    metrics = ['revenue', 'net_income', 'operating_income', 'eps', 'free_cash_flow', 'operating_cf']
    
    for metric in metrics:
        if metric in historical_data and len(historical_data[metric]) > 1:
            values = np.array(historical_data[metric])
            years = np.array(historical_data['years'])
            
            # Filter out zeros and invalid values
            valid_idx = values > 0
            if not any(valid_idx):
                continue
                
            values = values[valid_idx]
            years = years[valid_idx]
            
            if len(values) < 2:
                continue
            
            # 1. Simple Average Growth Rate
            pct_changes = np.diff(values) / values[:-1]
            growth_rates['simple_average'][metric] = np.mean(pct_changes)
            
            # 2. CAGR (Compound Annual Growth Rate)
            n_years = len(values) - 1
            cagr = (values[-1] / values[0]) ** (1/n_years) - 1
            growth_rates['cagr'][metric] = cagr
            
            # 3. Linear Regression Growth Rate
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
                avg_value = np.mean(values)
                regression_growth = slope / avg_value if avg_value != 0 else 0
                growth_rates['linear_regression'][metric] = regression_growth
            except:
                growth_rates['linear_regression'][metric] = growth_rates['cagr'][metric]
            
            # 4. Weighted Average (more weight to recent years)
            weights = np.linspace(0.5, 1.5, len(pct_changes))
            weighted_growth = np.average(pct_changes, weights=weights)
            growth_rates['weighted_average'][metric] = weighted_growth
            
            # 5. Volatility (Standard Deviation of growth rates)
            growth_rates['volatility'][metric] = np.std(pct_changes)
    
    return growth_rates


def project_financials_multi_method(historical_data, growth_rates, years=5, method='cagr'):
    """Project financial statements using specified methodology"""
    projections = {}
    base_year = historical_data['years'][-1]
    
    # Get base values (most recent year)
    base_values = {
        'revenue': historical_data['revenue'][-1] if historical_data['revenue'] else 0,
        'net_income': historical_data['net_income'][-1] if historical_data['net_income'] else 0,
        'operating_income': historical_data['operating_income'][-1] if historical_data['operating_income'] else 0,
        'eps': historical_data['eps'][-1] if historical_data['eps'] else 0,
        'free_cash_flow': historical_data['free_cash_flow'][-1] if historical_data['free_cash_flow'] else 0,
        'operating_cf': historical_data['operating_cf'][-1] if historical_data['operating_cf'] else 0
    }
    
    # Select growth rate method
    selected_growth = growth_rates.get(method, growth_rates.get('cagr', {}))
    
    for metric, base_value in base_values.items():
        if base_value and metric in selected_growth:
            growth_rate = selected_growth[metric]
            
            # Create projection including historical data
            historical_values = historical_data.get(metric, [])
            projection = {
                'historical_years': historical_data['years'],
                'historical_values': historical_values,
                'projected_years': [base_year + i for i in range(1, years + 1)],
                'projected_values': []
            }
            
            # Project forward
            for year in range(1, years + 1):
                projected_value = base_value * ((1 + growth_rate) ** year)
                projection['projected_values'].append(projected_value)
            
            projections[metric] = projection
    
    return projections


def monte_carlo_simulation(base_value, mean_growth, volatility, years=5, simulations=1000):
    """Run Monte Carlo simulation for projections"""
    results = []
    
    for _ in range(simulations):
        projection = [base_value]
        for year in range(years):
            # Random growth rate based on mean and volatility
            growth = np.random.normal(mean_growth, volatility)
            next_value = projection[-1] * (1 + growth)
            projection.append(next_value)
        results.append(projection[1:])  # Exclude base value
    
    results = np.array(results)
    
    # Calculate percentiles
    percentiles = {
        'p10': np.percentile(results, 10, axis=0),
        'p25': np.percentile(results, 25, axis=0),
        'p50': np.percentile(results, 50, axis=0),
        'p75': np.percentile(results, 75, axis=0),
        'p90': np.percentile(results, 90, axis=0),
        'mean': np.mean(results, axis=0)
    }
    
    return percentiles


def calculate_dcf_valuation(historical_data, growth_rates, stock_info, projection_years=5, ticker=None, industry_context=None):
    """Calculate DCF (Discounted Cash Flow) valuation with enhanced assumptions and transparency"""
    try:
        # Get required inputs - FCF is in millions
        fcf_current = historical_data['free_cash_flow'][-1] if historical_data['free_cash_flow'] else 0
        
        if fcf_current <= 0:
            st.warning("⚠️ DCF requires positive Free Cash Flow. Current FCF is negative or zero.")
            return None
        
        # Get current stock price - try multiple sources
        current_price = stock_info.get('currentPrice', 0)
        if current_price == 0:
            current_price = stock_info.get('regularMarketPrice', 0)
        if current_price == 0 and ticker:
            # Fetch fresh price data
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                current_price = stock.info.get('currentPrice', 0)
                if current_price == 0:
                    current_price = stock.info.get('regularMarketPrice', 0)
            except:
                pass
        
        # Calculate historical CapEx intensity for better projections
        capex_intensity = 0.06  # Default 6%
        if historical_data.get('capex') and historical_data.get('revenue'):
            capex_values = [c / r for c, r in zip(historical_data['capex'], historical_data['revenue']) if r > 0]
            if capex_values:
                capex_intensity = np.mean(capex_values)
        
        # Estimate WACC (Weighted Average Cost of Capital) with industry context
        beta = stock_info.get('beta', 1.0)
        if beta is None or beta == 0:
            beta = 1.0
        
        # Use current 10-year Treasury rate (update this periodically)
        risk_free_rate = 0.045  # 4.5% US 10-year treasury (Oct 2025)
        market_return = 0.10  # Historical market return ~10%
        equity_risk_premium = market_return - risk_free_rate
        
        # Cost of equity using CAPM
        cost_of_equity = risk_free_rate + beta * equity_risk_premium
        
        # Get actual debt and interest expense for cost of debt calculation
        total_debt = 0
        if historical_data.get('total_debt') and len(historical_data['total_debt']) > 0:
            total_debt = historical_data['total_debt'][-1]
        else:
            # Fallback: estimate from balance sheet
            total_assets = historical_data['total_assets'][-1] if historical_data['total_assets'] else 0
            total_equity = historical_data['total_equity'][-1] if historical_data['total_equity'] else 0
            total_debt = total_assets - total_equity if total_assets and total_equity else 0
            total_debt = max(0, total_debt)
        
        equity = historical_data['total_equity'][-1] if historical_data['total_equity'] else 1
        total_capital = total_debt + equity
        
        # Calculate actual cost of debt from interest expense if available
        cost_of_debt = 0.05  # Default 5%
        if historical_data.get('interest_expense') and len(historical_data['interest_expense']) > 0 and total_debt > 0:
            recent_interest = historical_data['interest_expense'][-1]
            cost_of_debt = recent_interest / total_debt if total_debt > 0 else 0.05
            cost_of_debt = max(0.02, min(0.12, cost_of_debt))  # Between 2% and 12%
        
        tax_rate = 0.21  # US corporate tax rate
        
        # Calculate WACC
        if total_capital > 0:
            wacc = (equity / total_capital) * cost_of_equity + (total_debt / total_capital) * cost_of_debt * (1 - tax_rate)
        else:
            wacc = cost_of_equity
        
        # Ensure WACC is reasonable
        wacc = max(0.06, min(0.18, wacc))  # Between 6% and 18%
        
        # Project FCF with validated growth rate
        fcf_growth_rate = growth_rates['cagr'].get('free_cash_flow', 0.08)
        
        # Additional validation: FCF growth should be reasonable
        revenue_growth = growth_rates['cagr'].get('revenue', 0.08)
        # FCF growth shouldn't wildly exceed revenue growth without good reason
        if fcf_growth_rate > revenue_growth * 1.5:
            fcf_growth_rate = revenue_growth * 1.2  # Allow 20% premium max
            st.info(f"📊 FCF growth adjusted to {fcf_growth_rate*100:.1f}% to align with revenue growth trends")
        
        # Cap FCF growth at reasonable levels based on industry
        max_fcf_growth = 0.20  # 20% default
        if industry_context:
            max_fcf_growth = industry_context.get('industry_growth_cap', 0.20)
        
        fcf_growth_rate = max(-0.10, min(max_fcf_growth, fcf_growth_rate))
        
        # Terminal growth rate - should be GDP-like
        terminal_growth_rate = 0.025  # 2.5% - long-term GDP growth
        
        # Industry adjustments for terminal growth
        if industry_context:
            sector = industry_context.get('sector', '')
            if sector in ['Utilities', 'Consumer Defensive', 'Real Estate']:
                terminal_growth_rate = 0.020  # 2% for mature industries
            elif sector in ['Technology', 'Healthcare']:
                terminal_growth_rate = 0.030  # 3% for growth industries
        
        # Ensure terminal growth < WACC (mathematical requirement)
        if terminal_growth_rate >= wacc:
            terminal_growth_rate = wacc * 0.6  # Set to 60% of WACC
        
        projected_fcf = []
        for year in range(1, projection_years + 1):
            fcf = fcf_current * ((1 + fcf_growth_rate) ** year)
            projected_fcf.append(fcf)
        
        # Calculate present value of projected FCF
        pv_fcf = []
        for i, fcf in enumerate(projected_fcf):
            pv = fcf / ((1 + wacc) ** (i + 1))
            pv_fcf.append(pv)
        
        # Calculate terminal value
        terminal_fcf = projected_fcf[-1] * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (wacc - terminal_growth_rate)
        pv_terminal_value = terminal_value / ((1 + wacc) ** projection_years)
        
        # Enterprise Value (in millions)
        enterprise_value = sum(pv_fcf) + pv_terminal_value
        
        # Equity Value (in millions)
        cash = stock_info.get('cash', 0) / 1e6 if stock_info.get('cash', 0) > 1e6 else stock_info.get('cash', 0)  # Convert to millions if in dollars
        # Use total_debt from earlier calculation, or get from stock_info
        debt_from_stockinfo = stock_info.get('totalDebt', 0) / 1e6 if stock_info.get('totalDebt', 0) > 1e6 else stock_info.get('totalDebt', total_debt)
        total_debt = debt_from_stockinfo if debt_from_stockinfo > 0 else total_debt  # Prefer stock_info, fallback to calculated
        
        equity_value = enterprise_value + cash - total_debt
        
        # Shares outstanding - get from stock_info or historical data
        shares_outstanding = stock_info.get('sharesOutstanding', 0)
        if shares_outstanding == 0 or shares_outstanding is None:
            # Try to estimate from historical data (in millions)
            shares_outstanding = historical_data.get('total_equity', [1])[-1] / 20 if historical_data.get('total_equity') else 1
            shares_outstanding = shares_outstanding * 1e6  # Convert to actual shares
        
        # Calculate fair value per share
        # equity_value is in millions, shares_outstanding is actual count
        fair_value_per_share = (equity_value * 1e6) / shares_outstanding if shares_outstanding > 0 else 0
        
        # Calculate upside/downside
        if current_price > 0 and fair_value_per_share > 0:
            upside = ((fair_value_per_share - current_price) / current_price) * 100
        else:
            upside = 0
        
        # Calculate sensitivity analysis
        sensitivity_analysis = calculate_dcf_sensitivity(
            projected_fcf, terminal_fcf, wacc, terminal_growth_rate, 
            projection_years, cash, total_debt, shares_outstanding
        )
        
        return {
            'fcf_current': fcf_current,
            'wacc': wacc,
            'cost_of_equity': cost_of_equity,
            'cost_of_debt': cost_of_debt,
            'fcf_growth_rate': fcf_growth_rate,
            'terminal_growth_rate': terminal_growth_rate,
            'projected_fcf': projected_fcf,
            'pv_fcf': pv_fcf,
            'terminal_value': terminal_value,
            'pv_terminal_value': pv_terminal_value,
            'enterprise_value': enterprise_value,
            'equity_value': equity_value,
            'fair_value_per_share': fair_value_per_share,
            'current_price': current_price,
            'upside': upside,
            'shares_outstanding': shares_outstanding,
            'debt': total_debt,
            'cash': cash,
            'beta': beta,
            'risk_free_rate': risk_free_rate,
            'equity_risk_premium': equity_risk_premium,
            'debt_to_equity': (total_debt / equity) if equity > 0 else 0,
            'capex_intensity': capex_intensity,
            'sensitivity_analysis': sensitivity_analysis
        }
    except Exception as e:
        st.warning(f"DCF calculation error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def calculate_dcf_sensitivity(projected_fcf, terminal_fcf, base_wacc, base_terminal_growth, projection_years, cash, debt, shares):
    """Calculate DCF sensitivity to key assumptions"""
    sensitivity = {}
    
    # WACC sensitivity (+/- 1%)
    wacc_scenarios = [base_wacc - 0.01, base_wacc, base_wacc + 0.01]
    wacc_values = []
    
    for wacc in wacc_scenarios:
        # Recalculate PV of FCF
        pv_fcf = sum([fcf / ((1 + wacc) ** (i + 1)) for i, fcf in enumerate(projected_fcf)])
        
        # Ensure terminal growth < WACC
        terminal_growth = min(base_terminal_growth, wacc * 0.6)
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        pv_terminal = terminal_value / ((1 + wacc) ** projection_years)
        
        ev = pv_fcf + pv_terminal
        equity_value = ev + cash - debt
        fair_value = (equity_value * 1e6) / shares if shares > 0 else 0
        wacc_values.append(fair_value)
    
    sensitivity['wacc'] = {
        'scenarios': [f'{w*100:.1f}%' for w in wacc_scenarios],
        'values': wacc_values
    }
    
    # Terminal growth sensitivity
    terminal_growth_scenarios = [
        max(0.015, base_terminal_growth - 0.005),
        base_terminal_growth,
        min(base_wacc * 0.6, base_terminal_growth + 0.005)
    ]
    terminal_values = []
    
    pv_fcf_base = sum([fcf / ((1 + base_wacc) ** (i + 1)) for i, fcf in enumerate(projected_fcf)])
    
    for tg in terminal_growth_scenarios:
        terminal_value = terminal_fcf / (base_wacc - tg)
        pv_terminal = terminal_value / ((1 + base_wacc) ** projection_years)
        ev = pv_fcf_base + pv_terminal
        equity_value = ev + cash - debt
        fair_value = (equity_value * 1e6) / shares if shares > 0 else 0
        terminal_values.append(fair_value)
    
    sensitivity['terminal_growth'] = {
        'scenarios': [f'{tg*100:.1f}%' for tg in terminal_growth_scenarios],
        'values': terminal_values
    }
    
    return sensitivity


def create_scenario_projections(historical_data, growth_rates, years=5):
    """Create Bull, Base, and Bear scenario projections"""
    scenarios = {}
    
    base_year = historical_data['years'][-1]
    
    metrics = ['revenue', 'net_income', 'eps', 'free_cash_flow']
    
    for metric in metrics:
        if metric not in growth_rates['cagr']:
            continue
            
        base_value = historical_data[metric][-1] if historical_data[metric] else 0
        base_growth = growth_rates['cagr'][metric]
        volatility = growth_rates['volatility'].get(metric, 0.1)
        
        # Bull case: higher growth (base + 1 std dev)
        bull_growth = base_growth + volatility
        
        # Base case: CAGR
        base_case_growth = base_growth
        
        # Bear case: lower growth (base - 1 std dev)
        bear_growth = base_growth - volatility
        
        scenarios[metric] = {
            'bull': {'growth_rate': bull_growth, 'values': []},
            'base': {'growth_rate': base_case_growth, 'values': []},
            'bear': {'growth_rate': bear_growth, 'values': []}
        }
        
        # Project each scenario
        for scenario_name, scenario_data in scenarios[metric].items():
            for year in range(1, years + 1):
                value = base_value * ((1 + scenario_data['growth_rate']) ** year)
                scenario_data['values'].append(value)
    
    return scenarios


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
    """Display advanced financial projections page with multiple methodologies"""
    st.subheader("🔮 Advanced Financial Projections & Valuation")
    st.markdown(f"Comprehensive financial analysis and valuation for **{ticker}** using multiple methodologies")
    
    # Handle None cached_info
    if cached_info is None:
        cached_info = {}
    
    # Advanced settings in expandable section
    with st.expander("⚙️ Projection Settings & Methodologies", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            projection_years = st.slider(
                "Projection Period (Years)",
                min_value=3,
                max_value=10,
                value=5,
                help="Number of years to project into the future"
            )
        
        with col2:
            projection_method = st.selectbox(
                "Primary Projection Method",
                options=['cagr', 'weighted_average', 'linear_regression', 'simple_average'],
                format_func=lambda x: {
                    'cagr': '📈 CAGR (Compound Annual Growth)',
                    'weighted_average': '⚖️ Weighted Average (Recent Focus)',
                    'linear_regression': '📊 Linear Regression',
                    'simple_average': '➗ Simple Average'
                }[x],
                help="Select the methodology for financial projections"
            )
        
        with col3:
            enable_monte_carlo = st.checkbox(
                "Enable Monte Carlo Simulation",
                value=True,
                help="Run probabilistic simulations with confidence intervals"
            )
        
        st.markdown("**Methodology Descriptions:**")
        st.markdown("""
        - **CAGR**: Compound Annual Growth Rate - smooths volatility, ideal for consistent growth
        - **Weighted Average**: Gives more weight to recent years - best for trend changes
        - **Linear Regression**: Statistical trend line - captures long-term trajectory
        - **Simple Average**: Equal weight to all years - baseline comparison
        - **Monte Carlo**: Probabilistic simulation considering volatility - provides confidence intervals
        """)
    
    # Fetch comprehensive historical data from 2020
    with st.spinner("📊 Loading comprehensive historical data (2020-2024) and analyzing trends..."):
        historical_data, growth_rates = calculate_historical_growth_rates(ticker, use_local_data=True)
    
    if historical_data is None or growth_rates is None:
        st.error("⚠️ Unable to fetch historical financial data for projections")
        st.info("💡 This feature requires access to detailed financial statements. Some companies may have limited data availability.")
        return
    
    # Display comprehensive historical performance (2020-2024)
    st.markdown("---")
    st.markdown("### 📈 Historical Financial Performance (2020-2024)")
    st.markdown("**Complete 5-Year Historical Data** - Foundation for all projections and valuation models")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get most recent values
    most_recent_year = historical_data['years'][-1]
    
    with col1:
        if historical_data['revenue']:
            latest_revenue = historical_data['revenue'][-1]
            revenue_cagr = growth_rates['cagr'].get('revenue', 0) * 100
            st.metric(
                f"Revenue ({most_recent_year})",
                f"${latest_revenue/1e3:.2f}B",
                f"{revenue_cagr:.1f}% CAGR (Historical)"
            )
    
    with col2:
        if historical_data['net_income']:
            latest_net_income = historical_data['net_income'][-1]
            ni_cagr = growth_rates['cagr'].get('net_income', 0) * 100
            st.metric(
                f"Net Income ({most_recent_year})",
                f"${latest_net_income/1e3:.2f}B",
                f"{ni_cagr:.1f}% CAGR (Historical)"
            )
    
    with col3:
        if historical_data['eps']:
            latest_eps = historical_data['eps'][-1]
            eps_cagr = growth_rates['cagr'].get('eps', 0) * 100
            st.metric(
                f"EPS ({most_recent_year})",
                f"${latest_eps:.2f}",
                f"{eps_cagr:.1f}% CAGR (Historical)"
            )
    
    with col4:
        if historical_data['free_cash_flow']:
            latest_fcf = historical_data['free_cash_flow'][-1]
            fcf_cagr = growth_rates['cagr'].get('free_cash_flow', 0) * 100
            st.metric(
                f"Free Cash Flow ({most_recent_year})",
                f"${latest_fcf/1e3:.2f}B",
                f"{fcf_cagr:.1f}% CAGR (Historical)"
            )
    
    st.caption(f"📅 Historical Data: 2020-{most_recent_year} | Data Source: Company Financials | Note: Metrics show historical CAGR - Projections below use **{projection_method.upper().replace('_', ' ')}** method")
    
    # Display comprehensive growth rates comparison
    st.markdown("---")
    st.markdown("### 📊 Growth Rate Analysis - Multiple Methodologies")
    st.markdown("Comparative growth rates using different calculation methods (2020-2024)")
    
    # Create comprehensive growth rate comparison table
    growth_comparison_data = []
    metrics = ['revenue', 'net_income', 'operating_income', 'eps', 'free_cash_flow']
    metric_names = ['Revenue', 'Net Income', 'Operating Income', 'EPS', 'Free Cash Flow']
    
    for metric, name in zip(metrics, metric_names):
        if metric in growth_rates['cagr']:
            growth_comparison_data.append({
                'Metric': name,
                'CAGR': f"{growth_rates['cagr'].get(metric, 0)*100:.2f}%",
                'Simple Avg': f"{growth_rates['simple_average'].get(metric, 0)*100:.2f}%",
                'Weighted Avg': f"{growth_rates['weighted_average'].get(metric, 0)*100:.2f}%",
                'Linear Reg': f"{growth_rates['linear_regression'].get(metric, 0)*100:.2f}%",
                'Volatility': f"{growth_rates['volatility'].get(metric, 0)*100:.2f}%",
                'Trend': '📈 Growing' if growth_rates['cagr'].get(metric, 0) > 0 else '📉 Declining'
            })
    
    if growth_comparison_data:
        growth_df = pd.DataFrame(growth_comparison_data)
        st.dataframe(growth_df, width='stretch', hide_index=True)
        
        st.caption("""
        **Methodology Comparison:** CAGR = Compound Annual Growth | Simple Avg = Arithmetic mean | 
        Weighted Avg = Recent years weighted higher | Linear Reg = Statistical trend | 
        Volatility = Standard deviation of growth rates
        """)
    
    # Generate projections using selected method
    st.markdown("---")
    st.markdown(f"### 🔮 Financial Projections ({projection_years}-Year Forecast)")
    st.markdown(f"**Method:** {projection_method.upper().replace('_', ' ')} | **Period:** 2020-{historical_data['years'][-1] + projection_years}")
    
    # Show the actual growth rates being used for projections
    if projection_method in growth_rates and 'revenue' in growth_rates[projection_method]:
        proj_revenue_growth = growth_rates[projection_method].get('revenue', 0) * 100
        proj_ni_growth = growth_rates[projection_method].get('net_income', 0) * 100
        proj_eps_growth = growth_rates[projection_method].get('eps', 0) * 100
        st.info(f"📊 **Using {projection_method.upper().replace('_', ' ')} Growth Rates:** Revenue: {proj_revenue_growth:.1f}% | Net Income: {proj_ni_growth:.1f}% | EPS: {proj_eps_growth:.1f}%")
    
    projections = project_financials_multi_method(historical_data, growth_rates, projection_years, projection_method)
    
    # Generate scenarios (Bull/Base/Bear)
    scenarios = create_scenario_projections(historical_data, growth_rates, projection_years)
    
    base_year = historical_data['years'][-1]
    
    # Create comprehensive projection charts with full historical data (2020-present)
    st.markdown("#### 📊 Historical Data (2020-2024) + Projections")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Revenue: Historical (2020-{base_year}) + Projected ({base_year+1}-{base_year+projection_years})',
            f'Net Income: Historical + Projected',
            f'EPS: Historical + Projected',
            f'Free Cash Flow: Historical + Projected'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Revenue projection with full historical context
    if 'revenue' in projections:
        proj_data = projections['revenue']
        # Historical data from 2020
        fig.add_trace(
            go.Scatter(
                x=proj_data['historical_years'],
                y=[v/1e3 for v in proj_data['historical_values']],  # Convert to billions
                name='Historical Revenue',
                line=dict(color='#1f77b4', width=3),
                mode='lines+markers',
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>Revenue: $%{y:.2f}B<extra></extra>'
            ),
            row=1, col=1
        )
        # Projected data
        fig.add_trace(
            go.Scatter(
                x=proj_data['projected_years'],
                y=[v/1e3 for v in proj_data['projected_values']],
                name='Projected Revenue',
                line=dict(color='#ff7f0e', width=3, dash='dash'),
                mode='lines+markers',
                marker=dict(size=8, symbol='diamond'),
                hovertemplate='<b>%{x}</b><br>Projected: $%{y:.2f}B<extra></extra>'
            ),
            row=1, col=1
        )
        # Add Monte Carlo confidence interval if enabled
        if enable_monte_carlo and 'revenue' in growth_rates['cagr']:
            base_value = proj_data['historical_values'][-1]
            mc_results = monte_carlo_simulation(
                base_value,
                growth_rates[projection_method]['revenue'],
                growth_rates['volatility'].get('revenue', 0.1),
                projection_years,
                1000
            )
            fig.add_trace(
                go.Scatter(
                    x=proj_data['projected_years'],
                    y=[v/1e3 for v in mc_results['p90']],
                    name='90th Percentile',
                    line=dict(color='lightblue', width=1, dash='dot'),
                    mode='lines',
                    showlegend=False,
                    hovertemplate='P90: $%{y:.2f}B<extra></extra>'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=proj_data['projected_years'],
                    y=[v/1e3 for v in mc_results['p10']],
                    name='10th Percentile',
                    line=dict(color='lightblue', width=1, dash='dot'),
                    fill='tonexty',
                    fillcolor='rgba(173, 216, 230, 0.2)',
                    mode='lines',
                    showlegend=False,
                    hovertemplate='P10: $%{y:.2f}B<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Net Income projection with full historical context
    if 'net_income' in projections:
        proj_data = projections['net_income']
        fig.add_trace(
            go.Scatter(
                x=proj_data['historical_years'],
                y=[v/1e3 for v in proj_data['historical_values']],
                name='Historical Net Income',
                line=dict(color='#2ca02c', width=3),
                mode='lines+markers',
                marker=dict(size=8),
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Net Income: $%{y:.2f}B<extra></extra>'
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=proj_data['projected_years'],
                y=[v/1e3 for v in proj_data['projected_values']],
                name='Projected Net Income',
                line=dict(color='#d62728', width=3, dash='dash'),
                mode='lines+markers',
                marker=dict(size=8, symbol='diamond'),
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Projected: $%{y:.2f}B<extra></extra>'
            ),
            row=1, col=2
        )
    
    # EPS projection with full historical context
    if 'eps' in projections:
        proj_data = projections['eps']
        fig.add_trace(
            go.Scatter(
                x=proj_data['historical_years'],
                y=proj_data['historical_values'],
                name='Historical EPS',
                line=dict(color='#9467bd', width=3),
                mode='lines+markers',
                marker=dict(size=8),
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>EPS: $%{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=proj_data['projected_years'],
                y=proj_data['projected_values'],
                name='Projected EPS',
                line=dict(color='#8c564b', width=3, dash='dash'),
                mode='lines+markers',
                marker=dict(size=8, symbol='diamond'),
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Projected: $%{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Free Cash Flow projection with full historical context
    if 'free_cash_flow' in projections:
        proj_data = projections['free_cash_flow']
        fig.add_trace(
            go.Scatter(
                x=proj_data['historical_years'],
                y=[v/1e3 for v in proj_data['historical_values']],
                name='Historical FCF',
                line=dict(color='#e377c2', width=3),
                mode='lines+markers',
                marker=dict(size=8),
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>FCF: $%{y:.2f}B<extra></extra>'
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=proj_data['projected_years'],
                y=[v/1e3 for v in proj_data['projected_values']],
                name='Projected FCF',
                line=dict(color='#7f7f7f', width=3, dash='dash'),
                mode='lines+markers',
                marker=dict(size=8, symbol='diamond'),
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>Projected: $%{y:.2f}B<extra></extra>'
            ),
            row=2, col=2
        )
    
    # Update axes labels
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    
    fig.update_yaxes(title_text="Revenue ($B)", row=1, col=1)
    fig.update_yaxes(title_text="Net Income ($B)", row=1, col=2)
    fig.update_yaxes(title_text="EPS ($)", row=2, col=1)
    fig.update_yaxes(title_text="Free Cash Flow ($B)", row=2, col=2)
    
    fig.update_layout(
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        template='plotly_white',
        title_text=f"{ticker} Comprehensive Financial Projections | Historical (2020-{base_year}) + Forecast ({base_year+1}-{base_year+projection_years})",
        title_x=0.5,
        title_font=dict(size=16)
    )
    
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': True, 'displaylogo': False})
    
    if enable_monte_carlo:
        st.caption("📊 **Chart includes Monte Carlo confidence intervals** (shaded blue area shows 10th-90th percentile range from 1000 simulations)")
    
    # Add projection assumptions and limitations
    with st.expander("⚠️ Projection Methodology & Important Limitations", expanded=False):
        st.markdown("""
        #### Projection Methodology
        
        Our projections use multiple validated approaches:
        
        1. **Historical Analysis (2020-2024)**: 5-year trend analysis across all key financial metrics
        2. **Growth Rate Validation**: All growth rates are capped at industry-reasonable levels based on:
           - Company size and maturity (market cap-based)
           - Industry/sector growth caps
           - Historical volatility
           - Recent trends (declining companies get conservative caps)
        3. **Multiple Methodologies**: CAGR, weighted average (recent focus), linear regression, simple average
        4. **Scenario Analysis**: Bull/Base/Bear cases using ±1 standard deviation from base growth
        5. **Monte Carlo Simulation**: 1,000 probabilistic scenarios incorporating historical volatility
        
        #### Key Limitations & Risk Factors
        
        **⚠️ Please be aware of these important limitations:**
        
        1. **Past Performance ≠ Future Results**: Historical growth rates may not continue due to:
           - Market saturation
           - Increased competition
           - Technological disruption
           - Economic cycles
           - Regulatory changes
        
        2. **Linear Projection Assumption**: Our models assume relatively smooth growth, but reality includes:
           - Unexpected shocks (recessions, pandemics, etc.)
           - Step-function changes (acquisitions, divestitures)
           - Market share shifts
           - Management changes
        
        3. **Industry Dynamics**: Even with industry caps, projections may not account for:
           - Disruptive competitors
           - Changing customer preferences
           - Supply chain disruptions
           - Geopolitical events
        
        4. **Margin Assumptions**: Projections assume relatively stable margins, which may not hold if:
           - Input costs change significantly
           - Pricing power erodes
           - Operating leverage changes
           - Mix shifts occur
        
        5. **Data Quality**: While we use comprehensive historical data, limitations include:
           - Accounting policy changes
           - One-time items not fully adjusted
           - Quality of estimates (e.g., CapEx approximations)
        
        **🎯 Best Practice**: Use these projections as ONE input in your analysis. Compare with:
        - Analyst consensus estimates
        - Company guidance
        - Industry benchmarks
        - Multiple valuation methods
        - Scenario planning
        """)
    
    # Display comprehensive projection table with scenarios
    st.markdown("---")
    st.markdown("### 📋 Detailed Projections Table - Scenario Analysis")
    st.markdown("**Bull/Base/Bear scenarios** based on historical volatility analysis")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 Base Case Projections", "🎯 Scenario Comparison"])
    
    with tab1:
        # Base case projections table
        table_data = []
        projection_years_list = [base_year] + [base_year + i for i in range(1, projection_years + 1)]
        
        for idx, year in enumerate(projection_years_list):
            row = {'Year': year}
            
            if 'revenue' in projections and idx == 0:
                row['Revenue ($B)'] = f"${projections['revenue']['historical_values'][-1]/1e3:.2f}"
            elif 'revenue' in projections and idx > 0:
                row['Revenue ($B)'] = f"${projections['revenue']['projected_values'][idx-1]/1e3:.2f}"
            
            if 'net_income' in projections and idx == 0:
                row['Net Income ($B)'] = f"${projections['net_income']['historical_values'][-1]/1e3:.2f}"
            elif 'net_income' in projections and idx > 0:
                row['Net Income ($B)'] = f"${projections['net_income']['projected_values'][idx-1]/1e3:.2f}"
            
            if 'eps' in projections and idx == 0:
                row['EPS ($)'] = f"${projections['eps']['historical_values'][-1]:.2f}"
            elif 'eps' in projections and idx > 0:
                row['EPS ($)'] = f"${projections['eps']['projected_values'][idx-1]:.2f}"
            
            if 'free_cash_flow' in projections and idx == 0:
                row['FCF ($B)'] = f"${projections['free_cash_flow']['historical_values'][-1]/1e3:.2f}"
            elif 'free_cash_flow' in projections and idx > 0:
                row['FCF ($B)'] = f"${projections['free_cash_flow']['projected_values'][idx-1]/1e3:.2f}"
            
            row['Type'] = 'Actual' if idx == 0 else 'Projected'
            table_data.append(row)
        
        projection_df = pd.DataFrame(table_data)
        st.dataframe(projection_df, width='stretch', hide_index=True)
    
    with tab2:
        # Scenario comparison
        st.markdown("**Revenue Scenarios** (Final Year Comparison)")
        scenario_data = []
        
        for metric in ['revenue', 'net_income', 'eps']:
            if metric in scenarios:
                metric_name = {'revenue': 'Revenue ($B)', 'net_income': 'Net Income ($B)', 'eps': 'EPS ($)'}[metric]
                divisor = 1e3 if metric != 'eps' else 1
                
                scenario_data.append({
                    'Metric': metric_name,
                    'Bear Case': f"${scenarios[metric]['bear']['values'][-1]/divisor:.2f}",
                    'Base Case': f"${scenarios[metric]['base']['values'][-1]/divisor:.2f}",
                    'Bull Case': f"${scenarios[metric]['bull']['values'][-1]/divisor:.2f}",
                    'Bear Growth': f"{scenarios[metric]['bear']['growth_rate']*100:.1f}%",
                    'Base Growth': f"{scenarios[metric]['base']['growth_rate']*100:.1f}%",
                    'Bull Growth': f"{scenarios[metric]['bull']['growth_rate']*100:.1f}%"
                })
        
        scenario_df = pd.DataFrame(scenario_data)
        st.dataframe(scenario_df, width='stretch', hide_index=True)
        st.caption(f"**Scenarios represent final year ({base_year + projection_years}) projections under different growth assumptions**")
    
    # DCF Valuation
    st.markdown("---")
    st.markdown("### 💰 DCF Valuation Analysis")
    st.markdown("**Discounted Cash Flow Model** - Intrinsic value estimation based on projected free cash flows")
    st.info("📊 **Enhanced DCF Model**: Uses historical CapEx intensity, actual cost of debt from interest expense, industry-adjusted terminal growth, and includes sensitivity analysis")
    
    # Get industry context for DCF
    industry_context = get_industry_context(ticker, cached_info)
    
    with st.spinner("Calculating DCF valuation..."):
        dcf_results = calculate_dcf_valuation(historical_data, growth_rates, cached_info, projection_years, ticker, industry_context)
    
    if dcf_results:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Fair Value (DCF)",
                f"${dcf_results['fair_value_per_share']:.2f}",
                help="Estimated fair value per share based on DCF model"
            )
        
        with col2:
            st.metric(
                "Current Price",
                f"${dcf_results['current_price']:.2f}",
                f"{dcf_results['upside']:.1f}% {'upside' if dcf_results['upside'] > 0 else 'downside'}",
                help="Current market price vs DCF fair value"
            )
        
        with col3:
            st.metric(
                "WACC",
                f"{dcf_results['wacc']*100:.2f}%",
                help="Weighted Average Cost of Capital - discount rate used"
            )
        
        with col4:
            st.metric(
                "Enterprise Value",
                f"${dcf_results['enterprise_value']/1e3:.2f}B",
                help="Total enterprise value from DCF"
            )
        
        # DCF Details in expander
        with st.expander("📊 DCF Model Details, Assumptions & Sensitivity Analysis", expanded=False):
            st.markdown("#### 🔑 Key Assumptions & Their Sources")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Cash Flow Assumptions:**")
                st.markdown(f"""
                - **Current FCF**: ${dcf_results['fcf_current']/1e3:.2f}B  
                  *From: Historical financial statements*
                - **FCF Growth Rate**: {dcf_results['fcf_growth_rate']*100:.2f}%  
                  *Based on: Historical CAGR, adjusted for industry norms*
                - **Terminal Growth**: {dcf_results['terminal_growth_rate']*100:.2f}%  
                  *Assumption: Long-term GDP growth + industry adjustment*
                - **CapEx Intensity**: {dcf_results.get('capex_intensity', 0.06)*100:.1f}%  
                  *From: {len(historical_data.get('capex', []))} years of historical data*
                """)
            
            with col2:
                st.markdown("**Cost of Capital (WACC):**")
                debt_to_equity = dcf_results.get('debt_to_equity', 0)
                st.markdown(f"""
                - **WACC**: {dcf_results['wacc']*100:.2f}%  
                  *Weighted average of debt and equity costs*
                - **Cost of Equity**: {dcf_results['cost_of_equity']*100:.2f}%  
                  *CAPM: Risk-free + Beta × Market premium*
                - **Cost of Debt**: {dcf_results.get('cost_of_debt', 0.05)*100:.2f}%  
                  *From: Interest expense / Total debt*
                - **Beta**: {dcf_results.get('beta', 1.0):.2f}  
                  *From: Yahoo Finance market data*
                - **Risk-Free Rate**: {dcf_results.get('risk_free_rate', 0.045)*100:.2f}%  
                  *US 10-Year Treasury*
                - **Equity Risk Premium**: {dcf_results.get('equity_risk_premium', 0.055)*100:.2f}%  
                  *Historical market premium*
                """)
            
            with col3:
                st.markdown("**Capital Structure:**")
                shares_m = dcf_results.get('shares_outstanding', 0) / 1e6
                st.markdown(f"""
                - **Total Debt**: ${dcf_results.get('debt', 0)/1e3:.2f}B
                - **Cash**: ${dcf_results.get('cash', 0)/1e3:.2f}B
                - **Net Debt**: ${(dcf_results.get('debt', 0) - dcf_results.get('cash', 0))/1e3:.2f}B
                - **Debt-to-Equity**: {debt_to_equity:.2f}x
                - **Shares Outstanding**: {shares_m:.0f}M
                """)
            
            st.markdown("---")
            st.markdown("#### 📊 Valuation Build-Up")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Enterprise Value Calculation:**")
                pv_fcf_total = sum(dcf_results['pv_fcf'])
                terminal_pct = (dcf_results['pv_terminal_value'] / dcf_results['enterprise_value']) * 100
                st.markdown(f"""
                - **PV of Projected FCF** ({projection_years} years): ${pv_fcf_total/1e3:.2f}B
                - **Terminal Value**: ${dcf_results['terminal_value']/1e3:.2f}B
                - **PV of Terminal Value**: ${dcf_results['pv_terminal_value']/1e3:.2f}B  
                  *({terminal_pct:.1f}% of Enterprise Value)*
                - **Enterprise Value**: ${dcf_results['enterprise_value']/1e3:.2f}B
                """)
                
                if terminal_pct > 80:
                    st.warning("⚠️ Terminal value represents >80% of valuation - DCF is highly sensitive to terminal assumptions")
            
            with col2:
                st.markdown("**Equity Value Calculation:**")
                st.markdown(f"""
                - **Enterprise Value**: ${dcf_results['enterprise_value']/1e3:.2f}B
                - **+ Cash & Equivalents**: ${dcf_results.get('cash', 0)/1e3:.2f}B
                - **- Total Debt**: ${dcf_results.get('debt', 0)/1e3:.2f}B
                - **= Equity Value**: ${dcf_results['equity_value']/1e3:.2f}B
                - **÷ Shares Outstanding**: {shares_m:.0f}M
                - **= Fair Value per Share**: ${dcf_results['fair_value_per_share']:.2f}
                """)
            
            st.markdown("---")
            st.markdown("#### 🎯 Sensitivity Analysis")
            st.markdown("See how the fair value changes with different assumptions:")
            
            if 'sensitivity_analysis' in dcf_results:
                sens = dcf_results['sensitivity_analysis']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**WACC Sensitivity (±1%)**")
                    wacc_df = pd.DataFrame({
                        'WACC': sens['wacc']['scenarios'],
                        'Fair Value': [f'${v:.2f}' for v in sens['wacc']['values']],
                        'Change': [
                            f"{((v - dcf_results['fair_value_per_share']) / dcf_results['fair_value_per_share'] * 100):+.1f}%" 
                            for v in sens['wacc']['values']
                        ]
                    })
                    st.dataframe(wacc_df, width='stretch', hide_index=True)
                    st.caption("Lower WACC = Higher valuation (less risky)")
                
                with col2:
                    st.markdown("**Terminal Growth Sensitivity (±0.5%)**")
                    tg_df = pd.DataFrame({
                        'Terminal Growth': sens['terminal_growth']['scenarios'],
                        'Fair Value': [f'${v:.2f}' for v in sens['terminal_growth']['values']],
                        'Change': [
                            f"{((v - dcf_results['fair_value_per_share']) / dcf_results['fair_value_per_share'] * 100):+.1f}%" 
                            for v in sens['terminal_growth']['values']
                        ]
                    })
                    st.dataframe(tg_df, width='stretch', hide_index=True)
                    st.caption("Higher terminal growth = Higher valuation")
            
            # FCF projection visualization
            fcf_years = list(range(1, projection_years + 1))
            fig_dcf = go.Figure()
            
            fig_dcf.add_trace(go.Bar(
                x=fcf_years,
                y=[fcf/1e3 for fcf in dcf_results['projected_fcf']],
                name='Projected FCF',
                marker_color='lightblue',
                text=[f"${fcf/1e3:.2f}B" for fcf in dcf_results['projected_fcf']],
                textposition='outside'
            ))
            
            fig_dcf.add_trace(go.Bar(
                x=fcf_years,
                y=[pv/1e3 for pv in dcf_results['pv_fcf']],
                name='Present Value of FCF',
                marker_color='darkblue',
                text=[f"${pv/1e3:.2f}B" for pv in dcf_results['pv_fcf']],
                textposition='outside'
            ))
            
            fig_dcf.update_layout(
                title="Free Cash Flow Projections & Present Values",
                xaxis_title="Year",
                yaxis_title="Value ($B)",
                barmode='group',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig_dcf, width='stretch', config={'displayModeBar': True, 'displaylogo': False})
            
            st.caption("**DCF Interpretation:** The DCF model estimates intrinsic value by discounting all future free cash flows to present value. A positive upside suggests the stock may be undervalued.")
    else:
        st.info("📊 DCF valuation requires positive free cash flow data. Ensure FCF metrics are available.")
    
    # Get analyst estimates
    st.markdown("---")
    st.markdown("### 🎯 Analyst Estimates & Market Expectations")
    
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
    from utils.text_utils import normalize_markdown_spacing
    from pathlib import Path
    
    # Initialize AI generator
    data_dir = Path("data")
    ai_generator = AIInsightsGenerator(data_dir)
    
    if ai_generator.enabled:
        with st.spinner("🧠 Generating comprehensive advanced projection analysis..."):
            # Prepare comprehensive context with all methodologies
            
            # Get final year projections
            final_year = base_year + projection_years
            final_revenue = projections['revenue']['projected_values'][-1]/1e3 if 'revenue' in projections else 0
            final_ni = projections['net_income']['projected_values'][-1]/1e3 if 'net_income' in projections else 0
            final_eps = projections['eps']['projected_values'][-1] if 'eps' in projections else 0
            final_fcf = projections['free_cash_flow']['projected_values'][-1]/1e3 if 'free_cash_flow' in projections else 0
            
            # Get scenario projections
            revenue_bull = scenarios['revenue']['bull']['values'][-1]/1e3 if 'revenue' in scenarios else 0
            revenue_bear = scenarios['revenue']['bear']['values'][-1]/1e3 if 'revenue' in scenarios else 0
            
            projection_summary = f"""
**Comprehensive Financial Projection Analysis for {ticker}:**

**Historical Performance (2020-{base_year}):**
- Revenue CAGR: {growth_rates['cagr'].get('revenue', 0)*100:.2f}%
- Net Income CAGR: {growth_rates['cagr'].get('net_income', 0)*100:.2f}%
- EPS CAGR: {growth_rates['cagr'].get('eps', 0)*100:.2f}%
- FCF CAGR: {growth_rates['cagr'].get('free_cash_flow', 0)*100:.2f}%
- Revenue Volatility: {growth_rates['volatility'].get('revenue', 0)*100:.2f}%

**Projection Methodology Comparison:**
- Primary Method Used: {projection_method.upper()}
- CAGR: {growth_rates['cagr'].get('revenue', 0)*100:.2f}%
- Weighted Average (recent focus): {growth_rates['weighted_average'].get('revenue', 0)*100:.2f}%
- Linear Regression: {growth_rates['linear_regression'].get('revenue', 0)*100:.2f}%
- Simple Average: {growth_rates['simple_average'].get('revenue', 0)*100:.2f}%

**Base Case Projections ({final_year}):**
- Revenue: ${final_revenue:.2f}B
- Net Income: ${final_ni:.2f}B
- EPS: ${final_eps:.2f}
- Free Cash Flow: ${final_fcf:.2f}B

**Scenario Analysis ({final_year}):**
- Bull Case Revenue: ${revenue_bull:.2f}B ({scenarios['revenue']['bull']['growth_rate']*100:.1f}% growth)
- Base Case Revenue: ${final_revenue:.2f}B ({growth_rates[projection_method]['revenue']*100:.1f}% growth)
- Bear Case Revenue: ${revenue_bear:.2f}B ({scenarios['revenue']['bear']['growth_rate']*100:.1f}% growth)
"""
            
            # Add DCF results if available
            dcf_context = ""
            if dcf_results:
                dcf_context = f"""
**DCF Valuation Results:**
- Fair Value per Share: ${dcf_results['fair_value_per_share']:.2f}
- Current Price: ${dcf_results['current_price']:.2f}
- Implied Upside/Downside: {dcf_results['upside']:.1f}%
- WACC (Discount Rate): {dcf_results['wacc']*100:.2f}%
- Enterprise Value: ${dcf_results['enterprise_value']/1e3:.2f}B
- FCF Growth Rate Used: {dcf_results['fcf_growth_rate']*100:.2f}%
- Terminal Growth Rate: {dcf_results['terminal_growth_rate']*100:.2f}%
"""
            
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
            prompt = f"""You are a senior financial analyst and equity research professional specializing in advanced financial modeling, valuation, and forecasting.

{projection_summary}

{dcf_context}

{market_context}

**Analysis Framework:**
This analysis uses multiple advanced methodologies:
1. **Historical Growth Analysis (2020-{base_year})**: 5-year CAGR analysis across all key metrics
2. **Multi-Method Projections**: CAGR, weighted average, linear regression, and simple average
3. **Scenario Analysis**: Bull/Base/Bear cases based on historical volatility
4. **DCF Valuation**: Intrinsic value based on discounted free cash flows
5. **Monte Carlo Simulation**: Probabilistic confidence intervals (if enabled)

**Visual Context:**
The comprehensive dashboard displays:
1. **Revenue Chart**: Full historical trend (2020-{base_year}) + {projection_years}-year projections with confidence intervals
2. **Net Income Chart**: Historical profitability trajectory + future projections
3. **EPS Chart**: Earnings per share path from 2020 through {final_year}
4. **Free Cash Flow Chart**: Historical FCF generation + projected cash flows (used in DCF)

**Your Task:**
Provide an advanced, comprehensive financial projection and valuation analysis (900-1200 words) in flowing, narrative paragraph format.

Write as a cohesive equity research narrative that:

1. **Opens with Historical Foundation & Trend Analysis**: Begin by analyzing the 5-year historical performance (2020-{base_year}), discussing the **{growth_rates['cagr'].get('revenue', 0)*100:.2f}% revenue CAGR** and **{growth_rates['volatility'].get('revenue', 0)*100:.2f}% volatility**. Establish whether these trends are sustainable and how the company's performance has evolved over this full period. Reference the complete historical data shown in all four charts.

2. **Evaluates Projection Methodologies**: Discuss how different projection methods (**{projection_method}** as primary, CAGR, weighted average, linear regression) produce varying growth estimates. Explain why the selected methodology is appropriate for {ticker} and how the **{growth_rates[projection_method].get('revenue', 0)*100:.2f}% revenue growth** compares to alternatives. Discuss the implications of the differences between methods.

3. **Analyzes Revenue & Profitability Projections**: Flow into detailed analysis of the base case projections showing revenue reaching **${final_revenue:.2f}B** and net income of **${final_ni:.2f}B** by {final_year}. Discuss margin trends, whether profitability is growing faster/slower than revenue, and what the historical charts (2020-{base_year}) suggest about future trajectory.

4. **Explores Scenario Analysis Framework**: Naturally incorporate the bull/base/bear scenario analysis. Discuss how the bull case (**${revenue_bull:.2f}B revenue**) vs bear case (**${revenue_bear:.2f}B**) scenarios bracket the possibilities. Explain what conditions would drive each scenario and which seems most probable based on industry dynamics and company-specific factors.

5. **Integrates DCF Valuation Analysis**: Weave in the DCF valuation showing **${dcf_results.get('fair_value_per_share', 0):.2f} fair value** vs **${dcf_results.get('current_price', 0):.2f} current price** (if DCF available). Discuss whether the **{dcf_results.get('upside', 0):.1f}% implied upside/downside** makes sense given the projected FCF growth of **{dcf_results.get('fcf_growth_rate', 0)*100:.2f}%** and the **{dcf_results.get('wacc', 0)*100:.2f}% WACC** discount rate. Evaluate if the DCF assumptions are reasonable.

6. **Compares with Market Expectations**: Compare our projections and DCF valuation with market expectations (analyst targets, forward P/E, PEG ratio). Explain any divergence between our analysis and market consensus. Discuss whether the market is pricing in expectations consistent with our base case, bull case, or bear case.

7. **Addresses Key Assumptions, Risks & Sensitivities**: Throughout the narrative, acknowledge critical assumptions (growth rates, WACC, terminal growth, margin stability) and what could cause actuals to differ materially. Discuss the **{growth_rates['volatility'].get('revenue', 0)*100:.2f}% historical volatility** and how this uncertainty is reflected in scenario and Monte Carlo analysis. Reference industry context and competitive dynamics.

8. **Synthesizes Valuation & Investment Thesis**: Build toward a comprehensive synthesis of all analysis—projections, scenarios, DCF, market comparison—to form an investment thesis. Discuss what the stock might be worth in {projection_years} years under different scenarios and whether current valuation offers attractive risk/reward.

9. **Concludes with Balanced Investment Recommendation**: End with a clear, balanced conclusion that ties together historical trends, projection analysis, valuation assessment, and scenario outcomes. Provide perspective on whether {ticker} represents an attractive investment opportunity given all the analysis presented.

**Formatting Requirements:**
- Write in flowing paragraphs (8-10 paragraphs total), NOT bullet points or multiple heading sections
- DO NOT use multiple heading levels (###) within your response - write as continuous prose
- Bold all key figures, growth rates, and metrics: **$50.2B revenue**, **15.3% CAGR**, **$125 fair value**
- Use smooth transitions between paragraphs to maintain narrative flow
- Write in an engaging, professional tone as if writing a comprehensive equity research report

Be highly specific, cite actual numbers from all projections/metrics/scenarios/DCF, reference all four projection charts showing 2020-{final_year} data naturally within the narrative, integrate multiple methodologies, and provide balanced, sophisticated analysis that a professional analyst would produce."""
            
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
                            "content": "You are a senior financial analyst and equity research professional with deep expertise in advanced financial modeling, DCF valuation, scenario analysis, and multi-method forecasting. Write in flowing, paragraph-based narrative format, NOT as bullet points or multiple sections with headings. Integrate multiple analytical frameworks (historical analysis, projection methodologies, scenario analysis, DCF valuation, market comparison) seamlessly into a cohesive narrative. Provide sophisticated, balanced analysis that synthesizes all methodologies and data points naturally within the narrative, as you would in a comprehensive professional equity research report. Reference all charts and data spanning 2020 to future projections throughout your analysis."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=2500
                )
                
                ai_insight = normalize_markdown_spacing(response.choices[0].message.content.strip())
                
                # Display AI insight using Streamlit's native components
                st.markdown("---")
                st.info("📈 AI-Generated Projection Analysis")
                st.markdown(ai_insight)
                st.markdown("---")
                
                # Add comprehensive disclaimer
                st.caption("""
                💡 **AI-Generated Advanced Financial Analysis**: This comprehensive analysis integrates multiple projection methodologies 
                (CAGR, weighted average, linear regression, simple average), scenario analysis (Bull/Base/Bear cases), 
                DCF valuation, and Monte Carlo simulation where applicable. Analysis is based on historical financial data (2020-2024), 
                current market metrics, and analyst expectations. Projections are estimates and may not reflect actual future performance. 
                The DCF valuation uses estimated WACC and growth assumptions that should be validated independently. 
                Always conduct additional research, validate assumptions, and consider multiple scenarios before making investment decisions.
                Past performance does not guarantee future results.
                """)
                
            except Exception as e:
                st.warning(f"⚠️ Unable to generate AI projection analysis: {str(e)}")
                st.info("Please ensure your OpenAI API key is properly configured in the .env file.")
    else:
        st.info("🔑 **AI Advanced Analysis Unavailable**: Configure your OpenAI API key in the .env file to enable comprehensive AI-powered analysis that synthesizes all projection methodologies, scenario analysis, DCF valuation, historical trends (2020-present), and market expectations into a cohesive investment thesis and recommendation.")
    
    # Final Summary and Important Disclosures
    st.markdown("---")
    st.markdown("### 📌 Key Takeaways & Critical Disclosures")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Strengths of This Analysis")
        st.markdown("""
        - **Comprehensive Historical Data**: 5-year trend analysis (2020-2024)
        - **Multiple Methodologies**: CAGR, weighted average, regression, simple average
        - **Industry-Validated Growth Rates**: Capped at reasonable industry benchmarks
        - **Scenario Analysis**: Bull/Base/Bear cases with probabilistic Monte Carlo
        - **Enhanced DCF Model**: Uses actual CapEx, debt costs, and industry-adjusted WACC
        - **Sensitivity Analysis**: Shows impact of key assumption changes
        - **Transparent Assumptions**: All inputs and calculations clearly documented
        """)
    
    with col2:
        st.markdown("#### ⚠️ Important Warnings & Limitations")
        st.markdown("""
        - **Not Financial Advice**: This is analytical tool output, not investment recommendations
        - **Historical Bias**: Past growth rates may not continue in different market conditions
        - **Model Assumptions**: Linear projections don't capture all real-world complexity
        - **Data Quality**: Estimates used where actual data unavailable (e.g., some CapEx)
        - **Market Changes**: Cannot predict disruption, competition, or macro shocks
        - **DCF Sensitivity**: Small assumption changes significantly impact valuation
        - **Requires Validation**: Cross-check with analyst consensus and company guidance
        """)
    
    st.markdown("---")
    st.error("""
    **⚠️ CRITICAL DISCLAIMER - PLEASE READ**
    
    This financial projection and valuation analysis is provided for informational and educational purposes only. 
    It is NOT investment advice, and should NOT be the sole basis for any investment decision.
    
    **Key Points:**
    - Projections are based on historical data and mathematical models with inherent limitations
    - Growth rates are capped at industry-reasonable levels, but no cap guarantees accuracy
    - DCF valuations are highly sensitive to assumptions (WACC, terminal growth, FCF growth)
    - Market conditions, competition, and company-specific factors can materially differ from projections
    - Past performance does not guarantee future results
    - All investments carry risk, including the potential loss of principal
    
    **Before Making Any Investment Decision:**
    1. Conduct your own thorough research
    2. Review company filings (10-K, 10-Q, 8-K)
    3. Compare with professional analyst research
    4. Consider your own risk tolerance and investment objectives
    5. Consult with a qualified financial advisor
    6. Validate all assumptions and calculations independently
    
    **Data Sources & Limitations:**
    - Historical data: Company financials via local CSV and Yahoo Finance
    - Market data: Yahoo Finance (may have delays or errors)
    - Some metrics estimated where actual data unavailable
    - Industry benchmarks based on general sector averages
    
    By using this analysis, you acknowledge that you understand these limitations and will not rely solely 
    on this information for investment decisions.
    """)
    
    st.caption(f"Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Ticker: {ticker} | Projection Period: {projection_years} years")
