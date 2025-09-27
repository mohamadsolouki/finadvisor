'use client'

import { useState, useEffect } from 'react'
import { DashboardLayout } from '@/components/layout/DashboardLayout'

interface StockData {
  symbol: string
  price: string
  change: string
  changePercent: string
  high: string
  low: string
  open: string
  previousClose: string
  volume: string
}

interface CompanyOverview {
  symbol: string
  name: string
  description: string
  industry: string
  sector: string
  marketCapitalization: string
  peRatio: string
  pegRatio: string
  bookValue: string
  dividendPerShare: string
  dividendYield: string
  eps: string
  revenuePerShareTTM: string
  profitMargin: string
  operatingMarginTTM: string
  returnOnAssetsTTM: string
  returnOnEquityTTM: string
  revenueTTM: string
  grossProfitTTM: string
}

export default function FinancialDataPage() {
  const [symbol, setSymbol] = useState('')
  const [stockData, setStockData] = useState<StockData | null>(null)
  const [companyData, setCompanyData] = useState<CompanyOverview | null>(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'quote' | 'overview'>('quote')

  const fetchFinancialData = async () => {
    if (!symbol.trim()) return

    setLoading(true)
    try {
      if (activeTab === 'quote') {
        const response = await fetch(`http://localhost:3001/api/financial-data/quote/${symbol.toUpperCase()}`)
        const data = await response.json()
        
        if (data.success) {
          setStockData(data.data)
        } else {
          console.error('Failed to fetch stock data:', data)
          setStockData(null)
        }
      } else {
        const response = await fetch(`http://localhost:3001/api/financial-data/overview/${symbol.toUpperCase()}`)
        const data = await response.json()
        
        if (data.success) {
          setCompanyData(data.data)
        } else {
          console.error('Failed to fetch company data:', data)
          setCompanyData(null)
        }
      }
    } catch (error) {
      console.error('Error fetching financial data:', error)
      setStockData(null)
      setCompanyData(null)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      fetchFinancialData()
    }
  }

  const formatNumber = (value: string) => {
    const num = parseFloat(value)
    if (isNaN(num)) return value
    return num.toLocaleString()
  }

  const formatCurrency = (value: string) => {
    const num = parseFloat(value)
    if (isNaN(num)) return value
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(num)
  }

  const formatMarketCap = (value: string) => {
    const num = parseFloat(value)
    if (isNaN(num)) return value
    
    if (num >= 1e12) return `$${(num / 1e12).toFixed(2)}T`
    if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`
    if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`
    return formatCurrency(value)
  }

  return (
    <DashboardLayout>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Financial Data</h1>
          <p className="text-gray-600 mt-2">Real-time stock quotes and company information</p>
        </div>

        {/* Search and Tabs */}
        <div className="bg-white rounded-lg shadow mb-6">
          <div className="p-6 border-b border-gray-200">
            <div className="flex gap-4 mb-4">
              <input
                type="text"
                placeholder="Enter stock symbol (e.g., AAPL, GOOGL, MSFT)..."
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                onKeyPress={handleKeyPress}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={fetchFinancialData}
                disabled={loading || !symbol.trim()}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {loading ? 'Loading...' : 'Get Data'}
              </button>
            </div>

            {/* Tabs */}
            <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
              <button
                onClick={() => setActiveTab('quote')}
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'quote'
                    ? 'bg-white text-blue-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Real-time Quote
              </button>
              <button
                onClick={() => setActiveTab('overview')}
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'overview'
                    ? 'bg-white text-blue-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Company Overview
              </button>
            </div>
          </div>
        </div>

        {/* Loading */}
        {loading && (
          <div className="text-center py-8">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <p className="mt-2 text-gray-600">Fetching financial data...</p>
          </div>
        )}

        {/* Stock Quote Data */}
        {activeTab === 'quote' && stockData && !loading && (
          <div className="bg-white rounded-lg shadow p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-gray-900">{stockData.symbol}</h2>
              <div className="flex items-center gap-4 mt-2">
                <span className="text-3xl font-bold text-gray-900">{formatCurrency(stockData.price)}</span>
                <span className={`text-lg font-semibold ${
                  parseFloat(stockData.change) >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {parseFloat(stockData.change) >= 0 ? '+' : ''}{formatCurrency(stockData.change)} ({stockData.changePercent})
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-500 mb-1">Open</h3>
                <p className="text-lg font-semibold text-gray-900">{formatCurrency(stockData.open)}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-500 mb-1">High</h3>
                <p className="text-lg font-semibold text-gray-900">{formatCurrency(stockData.high)}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-500 mb-1">Low</h3>
                <p className="text-lg font-semibold text-gray-900">{formatCurrency(stockData.low)}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-500 mb-1">Previous Close</h3>
                <p className="text-lg font-semibold text-gray-900">{formatCurrency(stockData.previousClose)}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg md:col-span-2 lg:col-span-4">
                <h3 className="text-sm font-medium text-gray-500 mb-1">Volume</h3>
                <p className="text-lg font-semibold text-gray-900">{formatNumber(stockData.volume)}</p>
              </div>
            </div>
          </div>
        )}

        {/* Company Overview Data */}
        {activeTab === 'overview' && companyData && !loading && (
          <div className="bg-white rounded-lg shadow p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-gray-900">{companyData.name} ({companyData.symbol})</h2>
              <div className="flex gap-4 mt-2">
                <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">{companyData.sector}</span>
                <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm">{companyData.industry}</span>
              </div>
            </div>

            {companyData.description && (
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Company Description</h3>
                <p className="text-gray-700 leading-relaxed">{companyData.description}</p>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-500 mb-1">Market Cap</h3>
                <p className="text-lg font-semibold text-gray-900">{formatMarketCap(companyData.marketCapitalization)}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-500 mb-1">P/E Ratio</h3>
                <p className="text-lg font-semibold text-gray-900">{companyData.peRatio || 'N/A'}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-500 mb-1">EPS</h3>
                <p className="text-lg font-semibold text-gray-900">${companyData.eps || 'N/A'}</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-500 mb-1">Dividend Yield</h3>
                <p className="text-lg font-semibold text-gray-900">{companyData.dividendYield || 'N/A'}%</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-500 mb-1">Profit Margin</h3>
                <p className="text-lg font-semibold text-gray-900">{companyData.profitMargin || 'N/A'}%</p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-500 mb-1">ROE</h3>
                <p className="text-lg font-semibold text-gray-900">{companyData.returnOnEquityTTM || 'N/A'}%</p>
              </div>
            </div>
          </div>
        )}

        {/* No Data State */}
        {!stockData && !companyData && !loading && symbol && (
          <div className="text-center py-8">
            <p className="text-gray-600">No data found for "{symbol}". Please check the symbol and try again.</p>
          </div>
        )}

        {/* Initial State */}
        {!symbol && (
          <div className="text-center py-12">
            <div className="text-gray-400 mb-4">
              <svg className="mx-auto h-16 w-16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">Get Financial Data</h3>
            <p className="text-gray-600">Enter a stock symbol to view real-time quotes and company information</p>
          </div>
        )}
      </div>
    </DashboardLayout>
  )
}