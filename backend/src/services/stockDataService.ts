import axios from 'axios';
import { PrismaClient } from '@prisma/client';
import { logger } from '../utils/logger';

const prisma = new PrismaClient();

export interface StockSearchResult {
  symbol: string;
  name: string;
  type: string;
  region: string;
  marketOpen: string;
  marketClose: string;
  timezone: string;
  currency: string;
  matchScore: string;
}

class StockDataService {
  private alphaVantageApiKey: string;
  private baseUrl = 'https://www.alphavantage.co/query';

  constructor() {
    this.alphaVantageApiKey = process.env.ALPHA_VANTAGE_API_KEY || '';
    if (!this.alphaVantageApiKey) {
      throw new Error('Alpha Vantage API key is required for stock data service. Please set ALPHA_VANTAGE_API_KEY environment variable.');
    }
  }

  async searchStocks(keywords: string): Promise<StockSearchResult[]> {
    try {
      const response = await axios.get(this.baseUrl, {
        params: {
          function: 'SYMBOL_SEARCH',
          keywords: keywords.trim(),
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

      const bestMatches = response.data['bestMatches'] || [];
      
      return bestMatches.map((match: any) => ({
        symbol: match['1. symbol'],
        name: match['2. name'],
        type: match['3. type'],
        region: match['4. region'],
        marketOpen: match['5. marketOpen'],
        marketClose: match['6. marketClose'],
        timezone: match['7. timezone'],
        currency: match['8. currency'],
        matchScore: match['9. matchScore']
      }));
    } catch (error) {
      logger.error(`Error searching stocks with keywords "${keywords}":`, error);
      throw new Error(`Failed to search stocks. Please ensure valid API key and try again.`);
    }
  }

  async getStockOverview(symbol: string) {
    try {
      const response = await axios.get(this.baseUrl, {
        params: {
          function: 'OVERVIEW',
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

      return response.data;
    } catch (error) {
      logger.error(`Error fetching overview for ${symbol}:`, error);
      throw new Error(`Failed to fetch stock overview for ${symbol}.`);
    }
  }

  async createOrUpdateStock(symbol: string): Promise<any> {
    try {
      // First check if stock exists in database
      const existingStock = await prisma.stock.findUnique({
        where: { symbol: symbol.toUpperCase() }
      });

      if (existingStock) {
        return existingStock;
      }

      // Fetch stock overview from Alpha Vantage
      const overview = await this.getStockOverview(symbol);

      // Extract relevant data
      const stockData = {
        symbol: overview.Symbol || symbol.toUpperCase(),
        name: overview.Name || `${symbol.toUpperCase()} Corporation`,
        exchange: overview.Exchange || 'Unknown',
        sector: overview.Sector || null,
        industry: overview.Industry || null,
        marketCap: overview.MarketCapitalization ? BigInt(overview.MarketCapitalization) : null,
        isActive: true
      };

      // Create stock in database
      const newStock = await prisma.stock.create({
        data: stockData
      });

      logger.info(`Created new stock entry for ${symbol.toUpperCase()}`);
      return newStock;
    } catch (error) {
      logger.error(`Error creating/updating stock ${symbol}:`, error);
      throw new Error(`Failed to create or update stock data for ${symbol}.`);
    }
  }

  async populateFinancialData(stockId: string, symbol: string): Promise<void> {
    try {
      // This method can be expanded to populate historical data, ratios, etc.
      // For now, we'll just ensure the stock exists
      logger.info(`Financial data population for ${symbol} can be implemented as needed`);
    } catch (error) {
      logger.error(`Error populating financial data for ${symbol}:`, error);
      throw new Error(`Failed to populate financial data for ${symbol}.`);
    }
  }
}

export const stockDataService = new StockDataService();