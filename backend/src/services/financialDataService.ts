import axios from 'axios';
import { PrismaClient } from '@prisma/client';
import { logger } from '../utils/logger';

const prisma = new PrismaClient();

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

export interface FinancialRatios {
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
}

class FinancialDataService {
  private alphaVantageApiKey: string;
  private baseUrl = 'https://www.alphavantage.co/query';

  constructor() {
    this.alphaVantageApiKey = process.env.ALPHA_VANTAGE_API_KEY || '';
    if (!this.alphaVantageApiKey) {
      logger.warn('Alpha Vantage API key not found. Financial data service will use mock data.');
    }
  }

  async getFinancialData(symbol: string, dataType: string, period: string): Promise<any> {
    try {
      if (!this.alphaVantageApiKey) {
        return this.getMockFinancialData(symbol, dataType);
      }

      // Check cache first
      const cachedData = await this.getCachedFinancialData(symbol, dataType, period);
      if (cachedData) {
        return cachedData;
      }

      let function_name = '';
      switch (dataType) {
        case 'overview':
          function_name = 'OVERVIEW';
          break;
        case 'income_statement':
          function_name = 'INCOME_STATEMENT';
          break;
        case 'balance_sheet':
          function_name = 'BALANCE_SHEET';
          break;
        case 'cash_flow':
          function_name = 'CASH_FLOW';
          break;
        default:
          function_name = 'OVERVIEW';
      }

      const response = await axios.get(this.baseUrl, {
        params: {
          function: function_name,
          symbol: symbol.toUpperCase(),
          apikey: this.alphaVantageApiKey
        },
        timeout: 10000
      });

      if (response.data['Error Message']) {
        throw new Error(`Alpha Vantage API Error: ${response.data['Error Message']}`);
      }

      if (response.data['Note']) {
        logger.warn('Alpha Vantage API rate limit reached, using cached or mock data');
        return this.getMockFinancialData(symbol, dataType);
      }

      // Cache the data
      await this.cacheFinancialData(symbol, dataType, period, response.data);

      return response.data;
    } catch (error) {
      logger.error(`Error fetching financial data for ${symbol}:`, error);
      // Return mock data as fallback
      return this.getMockFinancialData(symbol, dataType);
    }
  }

  async getQuote(symbol: string): Promise<Quote> {
    try {
      if (!this.alphaVantageApiKey) {
        return this.getMockQuote(symbol);
      }

      const response = await axios.get(this.baseUrl, {
        params: {
          function: 'GLOBAL_QUOTE',
          symbol: symbol.toUpperCase(),
          apikey: this.alphaVantageApiKey
        },
        timeout: 10000
      });

      const quote = response.data['Global Quote'];
      if (!quote) {
        return this.getMockQuote(symbol);
      }

      return {
        symbol: quote['01. symbol'],
        price: parseFloat(quote['05. price']),
        change: parseFloat(quote['09. change']),
        changePercent: parseFloat(quote['10. change percent'].replace('%', '')),
        volume: parseInt(quote['06. volume']),
        lastUpdated: quote['07. latest trading day']
      };
    } catch (error) {
      logger.error(`Error fetching quote for ${symbol}:`, error);
      return this.getMockQuote(symbol);
    }
  }

  async getHistoricalData(symbol: string, period: string, interval: string): Promise<HistoricalDataPoint[]> {
    try {
      if (!this.alphaVantageApiKey) {
        return this.getMockHistoricalData(symbol);
      }

      const function_name = interval === 'daily' ? 'TIME_SERIES_DAILY_ADJUSTED' : 'TIME_SERIES_INTRADAY';
      
      const response = await axios.get(this.baseUrl, {
        params: {
          function: function_name,
          symbol: symbol.toUpperCase(),
          apikey: this.alphaVantageApiKey,
          outputsize: period === '1M' ? 'compact' : 'full'
        },
        timeout: 15000
      });

      const timeSeriesKey = interval === 'daily' ? 'Time Series (Daily)' : `Time Series (${interval})`;
      const timeSeries = response.data[timeSeriesKey];

      if (!timeSeries) {
        return this.getMockHistoricalData(symbol);
      }

      const historicalData: HistoricalDataPoint[] = [];
      const dates = Object.keys(timeSeries).sort().reverse(); // Most recent first

      const limit = this.getPeriodLimit(period);
      const limitedDates = dates.slice(0, limit);

      for (const date of limitedDates) {
        const data = timeSeries[date];
        historicalData.push({
          date,
          open: parseFloat(data['1. open']),
          high: parseFloat(data['2. high']),
          low: parseFloat(data['3. low']),
          close: parseFloat(data['4. close']),
          adjustedClose: parseFloat(data['5. adjusted close'] || data['4. close']),
          volume: parseInt(data['6. volume'] || data['5. volume'])
        });
      }

      return historicalData;
    } catch (error) {
      logger.error(`Error fetching historical data for ${symbol}:`, error);
      return this.getMockHistoricalData(symbol);
    }
  }

  async getFinancialRatios(symbol: string, years: number): Promise<FinancialRatios[]> {
    try {
      // Try to get from database first
      const stock = await prisma.stock.findUnique({
        where: { symbol: symbol.toUpperCase() },
        include: {
          ratios: {
            orderBy: { year: 'desc' },
            take: years
          }
        }
      });

      if (stock && stock.ratios.length > 0) {
        return stock.ratios.map(ratio => ({
          symbol: stock.symbol,
          year: ratio.year,
          peRatio: ratio.peRatio,
          pbRatio: ratio.pbRatio,
          debtToEquity: ratio.debtToEquity,
          currentRatio: ratio.currentRatio,
          quickRatio: ratio.quickRatio,
          returnOnEquity: ratio.returnOnEquity,
          returnOnAssets: ratio.returnOnAssets,
          grossMargin: ratio.grossMargin,
          operatingMargin: ratio.operatingMargin,
          netMargin: ratio.netMargin,
          dividendYield: ratio.dividendYield
        }));
      }

      // Fallback to mock data
      return this.getMockFinancialRatios(symbol, years);
    } catch (error) {
      logger.error(`Error fetching financial ratios for ${symbol}:`, error);
      return this.getMockFinancialRatios(symbol, years);
    }
  }

  async getEarnings(symbol: string): Promise<any> {
    try {
      if (!this.alphaVantageApiKey) {
        return this.getMockEarnings(symbol);
      }

      const response = await axios.get(this.baseUrl, {
        params: {
          function: 'EARNINGS',
          symbol: symbol.toUpperCase(),
          apikey: this.alphaVantageApiKey
        },
        timeout: 10000
      });

      return response.data;
    } catch (error) {
      logger.error(`Error fetching earnings for ${symbol}:`, error);
      return this.getMockEarnings(symbol);
    }
  }

  private async getCachedFinancialData(symbol: string, dataType: string, period: string): Promise<any> {
    try {
      const stock = await prisma.stock.findUnique({
        where: { symbol: symbol.toUpperCase() },
        include: {
          financialData: {
            where: {
              dataType,
              period
            },
            orderBy: { updatedAt: 'desc' },
            take: 1
          }
        }
      });

      if (stock && stock.financialData.length > 0) {
        const data = stock.financialData[0];
        const hoursSinceUpdate = (Date.now() - data.updatedAt.getTime()) / (1000 * 60 * 60);
        
        // Return cached data if it's less than 24 hours old
        if (hoursSinceUpdate < 24) {
          return {
            revenue: data.revenue?.toString(),
            netIncome: data.netIncome?.toString(),
            totalAssets: data.totalAssets?.toString(),
            eps: data.eps,
            cached: true
          };
        }
      }

      return null;
    } catch (error) {
      logger.error('Error getting cached financial data:', error);
      return null;
    }
  }

  private async cacheFinancialData(symbol: string, dataType: string, period: string, data: any): Promise<void> {
    try {
      const stock = await prisma.stock.findUnique({
        where: { symbol: symbol.toUpperCase() }
      });

      if (!stock) return;

      await prisma.financialData.upsert({
        where: {
          stockId_period_dataType: {
            stockId: stock.id,
            period,
            dataType
          }
        },
        update: {
          // Update with parsed data from API response
          updatedAt: new Date()
        },
        create: {
          stockId: stock.id,
          period,
          dataType,
          // Parse and store relevant fields from data
        }
      });
    } catch (error) {
      logger.error('Error caching financial data:', error);
    }
  }

  private getPeriodLimit(period: string): number {
    switch (period) {
      case '1M': return 30;
      case '3M': return 90;
      case '6M': return 180;
      case '1Y': return 365;
      case '2Y': return 730;
      case '5Y': return 1825;
      default: return 365;
    }
  }

  // Mock data methods for development and fallback
  private getMockFinancialData(symbol: string, dataType: string): any {
    return {
      Symbol: symbol.toUpperCase(),
      Name: `${symbol.toUpperCase()} Corporation`,
      Exchange: 'NASDAQ',
      Sector: 'Technology',
      Industry: 'Software',
      MarketCapitalization: '1000000000000',
      PERatio: '25.5',
      EPS: '12.34',
      DividendYield: '1.5%',
      Beta: '1.2',
      mock: true
    };
  }

  private getMockQuote(symbol: string): Quote {
    const basePrice = 150 + Math.random() * 100;
    const change = (Math.random() - 0.5) * 10;
    
    return {
      symbol: symbol.toUpperCase(),
      price: Math.round(basePrice * 100) / 100,
      change: Math.round(change * 100) / 100,
      changePercent: Math.round((change / basePrice) * 10000) / 100,
      volume: Math.floor(Math.random() * 10000000) + 1000000,
      lastUpdated: new Date().toISOString().split('T')[0]
    };
  }

  private getMockHistoricalData(symbol: string): HistoricalDataPoint[] {
    const data: HistoricalDataPoint[] = [];
    const basePrice = 150;
    let currentPrice = basePrice;
    
    for (let i = 365; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      
      const volatility = 0.02; // 2% daily volatility
      const change = (Math.random() - 0.5) * volatility * currentPrice * 2;
      
      const open = currentPrice;
      const close = Math.max(1, currentPrice + change);
      const high = Math.max(open, close) * (1 + Math.random() * 0.01);
      const low = Math.min(open, close) * (1 - Math.random() * 0.01);
      
      data.push({
        date: date.toISOString().split('T')[0],
        open: Math.round(open * 100) / 100,
        high: Math.round(high * 100) / 100,
        low: Math.round(low * 100) / 100,
        close: Math.round(close * 100) / 100,
        adjustedClose: Math.round(close * 100) / 100,
        volume: Math.floor(Math.random() * 10000000) + 1000000
      });
      
      currentPrice = close;
    }
    
    return data.reverse();
  }

  private getMockFinancialRatios(symbol: string, years: number): FinancialRatios[] {
    const ratios: FinancialRatios[] = [];
    const currentYear = new Date().getFullYear();
    
    for (let i = 0; i < years; i++) {
      ratios.push({
        symbol: symbol.toUpperCase(),
        year: currentYear - i,
        peRatio: 20 + Math.random() * 20,
        pbRatio: 3 + Math.random() * 5,
        debtToEquity: 0.5 + Math.random() * 1.5,
        currentRatio: 1 + Math.random() * 2,
        quickRatio: 0.8 + Math.random() * 1.5,
        returnOnEquity: 10 + Math.random() * 20,
        returnOnAssets: 5 + Math.random() * 15,
        grossMargin: 30 + Math.random() * 40,
        operatingMargin: 15 + Math.random() * 25,
        netMargin: 10 + Math.random() * 20,
        dividendYield: Math.random() * 5
      });
    }
    
    return ratios;
  }

  private getMockEarnings(symbol: string): any {
    return {
      symbol: symbol.toUpperCase(),
      annualEarnings: [
        { fiscalDateEnding: '2023-12-31', reportedEPS: '12.34' },
        { fiscalDateEnding: '2022-12-31', reportedEPS: '11.05' },
        { fiscalDateEnding: '2021-12-31', reportedEPS: '9.87' }
      ],
      quarterlyEarnings: [
        { fiscalDateEnding: '2023-12-31', reportedEPS: '3.45', estimatedEPS: '3.40', surprise: '0.05' },
        { fiscalDateEnding: '2023-09-30', reportedEPS: '3.12', estimatedEPS: '3.15', surprise: '-0.03' }
      ]
    };
  }
}

export const financialDataService = new FinancialDataService();