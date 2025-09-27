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
      throw new Error('Alpha Vantage API key is required for financial data service. Please set ALPHA_VANTAGE_API_KEY environment variable.');
    }
  }

  async getFinancialData(symbol: string, dataType: string, period: string): Promise<any> {
    try {
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
        throw new Error('Alpha Vantage API rate limit reached. Please try again later.');
      }

      // Cache the data
      await this.cacheFinancialData(symbol, dataType, period, response.data);

      return response.data;
    } catch (error) {
      logger.error(`Error fetching financial data for ${symbol}:`, error);
      throw new Error(`Failed to fetch financial data for ${symbol}. Please ensure valid API key and symbol.`);
    }
  }

  async getQuote(symbol: string): Promise<Quote> {
    try {

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
        throw new Error(`No quote data available for symbol ${symbol}`);
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
      throw new Error(`Failed to fetch quote for ${symbol}. Please ensure valid API key and symbol.`);
    }
  }

  async getHistoricalData(symbol: string, period: string, interval: string): Promise<HistoricalDataPoint[]> {
    try {

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
        throw new Error(`No historical data available for symbol ${symbol}`);
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
      throw new Error(`Failed to fetch historical data for ${symbol}. Please ensure valid API key and symbol.`);
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

      // No ratios available in database
      throw new Error(`No financial ratios available for symbol ${symbol}. Data needs to be populated from financial statements.`);
    } catch (error) {
      logger.error(`Error fetching financial ratios for ${symbol}:`, error);
      throw new Error(`Failed to fetch financial ratios for ${symbol}.`);
    }
  }

  async getEarnings(symbol: string): Promise<any> {
    try {

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
      throw new Error(`Failed to fetch earnings for ${symbol}. Please ensure valid API key and symbol.`);
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


}

export const financialDataService = new FinancialDataService();