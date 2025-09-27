'use client'

import { useState, useEffect } from 'react'
import { DashboardLayout } from '@/components/layout/DashboardLayout'

interface Stock {
  symbol: string
  name: string
  type: string
  region: string
  marketOpen: string
  marketClose: string
  timezone: string
  currency: string
  matchScore: string
}

export default function StocksPage() {
  const [searchTerm, setSearchTerm] = useState('')
  const [stocks, setStocks] = useState<Stock[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedStocks, setSelectedStocks] = useState<string[]>([])

  const searchStocks = async () => {
    if (!searchTerm.trim()) return
    
    setLoading(true)
    try {
      const response = await fetch(`http://localhost:3001/api/stocks/search?query=${encodeURIComponent(searchTerm)}`)
      const data = await response.json()
      
      if (data.success && data.data.bestMatches) {
        setStocks(data.data.bestMatches)
      } else {
        console.error('Failed to fetch stocks:', data)
        setStocks([])
      }
    } catch (error) {
      console.error('Error searching stocks:', error)
      setStocks([])
    } finally {
      setLoading(false)
    }
  }

  const toggleStockSelection = (symbol: string) => {
    setSelectedStocks(prev => 
      prev.includes(symbol) 
        ? prev.filter(s => s !== symbol)
        : [...prev, symbol]
    )
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      searchStocks()
    }
  }

  return (
    <DashboardLayout>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Stock Selection</h1>
          <p className="text-gray-600 mt-2">Search and select stocks for analysis</p>
        </div>

        {/* Search Section */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <div className="flex gap-4 mb-4">
            <input
              type="text"
              placeholder="Search for stocks (e.g., AAPL, Microsoft, Tesla)..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onKeyPress={handleKeyPress}
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <button
              onClick={searchStocks}
              disabled={loading || !searchTerm.trim()}
              className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
          </div>
          
          {selectedStocks.length > 0 && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Selected Stocks ({selectedStocks.length}):</h3>
              <div className="flex flex-wrap gap-2">
                {selectedStocks.map(symbol => (
                  <span key={symbol} className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
                    {symbol}
                    <button 
                      onClick={() => toggleStockSelection(symbol)}
                      className="ml-2 text-blue-600 hover:text-blue-800"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Results Section */}
        {loading && (
          <div className="text-center py-8">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <p className="mt-2 text-gray-600">Searching stocks...</p>
          </div>
        )}

        {stocks.length > 0 && (
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">Search Results</h2>
            </div>
            <div className="divide-y divide-gray-200">
              {stocks.map((stock, index) => (
                <div key={index} className="p-6 hover:bg-gray-50">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-4 mb-2">
                        <h3 className="text-lg font-semibold text-gray-900">{stock.symbol}</h3>
                        <span className="bg-gray-100 text-gray-700 px-2 py-1 rounded text-sm">{stock.type}</span>
                        <span className="text-sm text-gray-500">Match: {(parseFloat(stock.matchScore) * 100).toFixed(0)}%</span>
                      </div>
                      <p className="text-gray-700 mb-2">{stock.name}</p>
                      <div className="text-sm text-gray-500 space-y-1">
                        <p>Region: {stock.region} | Currency: {stock.currency}</p>
                        <p>Market Hours: {stock.marketOpen} - {stock.marketClose} ({stock.timezone})</p>
                      </div>
                    </div>
                    <div className="ml-4">
                      <button
                        onClick={() => toggleStockSelection(stock.symbol)}
                        className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                          selectedStocks.includes(stock.symbol)
                            ? 'bg-green-600 text-white hover:bg-green-700'
                            : 'bg-blue-600 text-white hover:bg-blue-700'
                        }`}
                      >
                        {selectedStocks.includes(stock.symbol) ? 'Selected ✓' : 'Select'}
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {stocks.length === 0 && searchTerm && !loading && (
          <div className="text-center py-8">
            <p className="text-gray-600">No stocks found for "{searchTerm}". Try a different search term.</p>
          </div>
        )}

        {!searchTerm && (
          <div className="text-center py-12">
            <div className="text-gray-400 mb-4">
              <svg className="mx-auto h-16 w-16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">Search for Stocks</h3>
            <p className="text-gray-600">Enter a company name or stock symbol to get started</p>
          </div>
        )}
      </div>
    </DashboardLayout>
  )
}