'use client'

import { ReactNode } from 'react'

interface DashboardLayoutProps {
  children: ReactNode
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-blue-600">FinAdvisor</h1>
            </div>
            <nav className="flex space-x-8">
              <a href="/dashboard" className="text-gray-700 hover:text-blue-600">Dashboard</a>
              <a href="/stocks" className="text-gray-700 hover:text-blue-600">Stocks</a>
              <a href="/analysis" className="text-gray-700 hover:text-blue-600">Analysis</a>
              <a href="/chat" className="text-gray-700 hover:text-blue-600">AI Chat</a>
            </nav>
          </div>
        </div>
      </header>
      <main>{children}</main>
    </div>
  )
}