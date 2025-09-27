// User types
export interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  role: 'USER' | 'ADMIN';
  createdAt: string;
  updatedAt: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
}

export interface AuthResponse {
  token: string;
  user: User;
  message: string;
}

// Stock types
export interface Stock {
  id: string;
  symbol: string;
  name: string;
  exchange: string;
  sector?: string;
  industry?: string;
  marketCap?: number;
  isActive: boolean;
}

export interface UserStock {
  id: string;
  symbol: string;
  name: string;
  exchange: string;
  sector?: string;
  industry?: string;
  marketCap?: number;
  selectedAt: string;
  notes?: string;
}

export interface StockSearchParams {
  q?: string;
  sector?: string;
  exchange?: string;
  limit?: number;
}

// Financial data types
export interface Quote {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  avgVolume?: number;
  marketCap?: number;
  peRatio?: number;
  lastUpdated: string;
}

export interface HistoricalDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  adjustedClose?: number;
  volume: number;
}

export interface FinancialRatio {
  symbol: string;
  year: number;
  peRatio?: number;
  pbRatio?: number;
  debtToEquity?: number;
  currentRatio?: number;
  quickRatio?: number;
  returnOnEquity?: number;
  returnOnAssets?: number;
  grossMargin?: number;
  operatingMargin?: number;
  netMargin?: number;
  dividendYield?: number;
  priceToSales?: number;
  priceToBook?: number;
  evToEbitda?: number;
}

export interface FinancialOverview {
  symbol: string;
  name: string;
  exchange: string;
  sector: string;
  industry: string;
  marketCap: string;
  peRatio: string;
  eps: string;
  dividendYield: string;
  beta: string;
  fiftyTwoWeekHigh: string;
  fiftyTwoWeekLow: string;
  description: string;
}

// Chat types
export interface ChatMessage {
  id: string;
  message: string;
  response: string;
  createdAt: string;
  tokensUsed?: number;
}

export interface ChatRequest {
  message: string;
  context?: any;
  selectedStocks?: string[];
}

export interface ChatResponse {
  response: string;
  context: any;
  timestamp: string;
  tokensUsed?: number;
}

export interface AnalysisRequest {
  symbols: string[];
  analysisType: 'overview' | 'detailed' | 'comparison';
}

export interface StockAnalysis {
  analysis: string;
  symbols: string[];
  analysisType: string;
  stocksData: any[];
  timestamp: string;
}

// API Response types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
  errors?: any[];
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page?: number;
  limit?: number;
  hasMore?: boolean;
}

// Chart data types
export interface ChartDataPoint {
  x: string | number;
  y: number;
  label?: string;
}

export interface ChartDataset {
  label: string;
  data: ChartDataPoint[];
  color?: string;
  backgroundColor?: string;
  borderColor?: string;
}

// Utility types
export type SortDirection = 'asc' | 'desc';
export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export interface SortConfig {
  key: string;
  direction: SortDirection;
}

export interface FilterConfig {
  [key: string]: any;
}

// Market data types
export interface MarketSummary {
  indices: {
    name: string;
    value: number;
    change: number;
    changePercent: number;
  }[];
  topGainers: {
    symbol: string;
    name: string;
    price: number;
    changePercent: number;
  }[];
  topLosers: {
    symbol: string;
    name: string;
    price: number;
    changePercent: number;
  }[];
  mostActive: {
    symbol: string;
    name: string;
    volume: number;
    price: number;
  }[];
}

// Notification types
export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  actionUrl?: string;
}

// Dashboard types
export interface DashboardMetrics {
  totalStocks: number;
  portfolioValue: number;
  dayChange: number;
  dayChangePercent: number;
  weekChange: number;
  weekChangePercent: number;
  monthChange: number;
  monthChangePercent: number;
}

export interface WatchlistItem {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  alerts?: {
    type: 'price' | 'volume';
    condition: 'above' | 'below';
    value: number;
  }[];
}