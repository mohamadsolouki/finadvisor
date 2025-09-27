'use client'

import { ReactNode } from 'react'
import { AuthProvider } from '@/stores/authStore'
import { StockProvider } from '@/stores/stockStore'

interface ProvidersProps {
  children: ReactNode
}

export function Providers({ children }: ProvidersProps) {
  return (
    <AuthProvider>
      <StockProvider>
        {children}
      </StockProvider>
    </AuthProvider>
  )
}