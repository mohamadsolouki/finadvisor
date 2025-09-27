'use client'

import { createContext, useContext, useState, ReactNode } from 'react'

interface Stock {
  id: string
  symbol: string
  name: string
  price?: number
  change?: number
  changePercent?: number
}

interface StockContextType {
  selectedStocks: Stock[]
  addStock: (stock: Stock) => void
  removeStock: (symbol: string) => void
  searchResults: Stock[]
  searchStocks: (query: string) => Promise<void>
  loading: boolean
}

const StockContext = createContext<StockContextType | undefined>(undefined)

export function StockProvider({ children }: { children: ReactNode }) {
  const [selectedStocks, setSelectedStocks] = useState<Stock[]>([])
  const [searchResults, setSearchResults] = useState<Stock[]>([])
  const [loading, setLoading] = useState(false)

  const addStock = (stock: Stock) => {
    setSelectedStocks(prev => {
      if (prev.find(s => s.symbol === stock.symbol)) {
        return prev
      }
      return [...prev, stock]
    })
  }

  const removeStock = (symbol: string) => {
    setSelectedStocks(prev => prev.filter(s => s.symbol !== symbol))
  }

  const searchStocks = async (query: string) => {
    if (!query.trim()) {
      setSearchResults([])
      return
    }

    setLoading(true)
    try {
      const token = localStorage.getItem('token')
      const response = await fetch(`http://localhost:3001/api/stocks/search?q=${encodeURIComponent(query)}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        setSearchResults(data)
      } else {
        setSearchResults([])
      }
    } catch (error) {
      console.error('Stock search failed:', error)
      setSearchResults([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <StockContext.Provider value={{
      selectedStocks,
      addStock,
      removeStock,
      searchResults,
      searchStocks,
      loading
    }}>
      {children}
    </StockContext.Provider>
  )
}

export function useStock() {
  const context = useContext(StockContext)
  if (context === undefined) {
    throw new Error('useStock must be used within a StockProvider')
  }
  return context
}