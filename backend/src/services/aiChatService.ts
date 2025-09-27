import OpenAI from 'openai';
import { PrismaClient } from '@prisma/client';
import { logger } from '../utils/logger';
import { financialDataService } from './financialDataService';

const prisma = new PrismaClient();

export interface ChatRequest {
  userId: string;
  message: string;
  context?: any;
  selectedStocks?: string[];
}

export interface ChatResponse {
  message: string;
  context: any;
  tokensUsed?: number;
}

export interface AnalysisRequest {
  userId: string;
  symbols: string[];
  analysisType: 'overview' | 'detailed' | 'comparison';
}

class AIChatService {
  private openai: OpenAI;
  private model: string = 'gpt-4.1-nano';

  constructor() {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error('OpenAI API key is required');
    }

    this.openai = new OpenAI({
      apiKey
    });
  }

  async generateResponse(request: ChatRequest): Promise<ChatResponse> {
    try {
      // Get financial context for selected stocks
      const financialContext = await this.getFinancialContext(request.selectedStocks || []);
      
      // Build system prompt
      const systemPrompt = this.buildSystemPrompt(financialContext);
      
      // Get recent chat history for context
      const chatHistory = await this.getRecentChatHistory(request.userId, 5);
      
      // Build messages array
      const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
        { role: 'system', content: systemPrompt }
      ];

      // Add chat history
      chatHistory.forEach(chat => {
        messages.push(
          { role: 'user', content: chat.message },
          { role: 'assistant', content: chat.response }
        );
      });

      // Add current message
      messages.push({ role: 'user', content: request.message });

      // Generate response
      const completion = await this.openai.chat.completions.create({
        model: this.model,
        messages,
        max_tokens: 1000,
        temperature: 0.7,
        presence_penalty: 0.1,
        frequency_penalty: 0.1
      });

      const response = completion.choices[0]?.message?.content || 'I apologize, but I could not generate a response.';
      const tokensUsed = completion.usage?.total_tokens || 0;

      // Save to chat history
      await this.saveChatHistory({
        userId: request.userId,
        message: request.message,
        response,
        context: { 
          ...request.context,
          selectedStocks: request.selectedStocks,
          tokensUsed
        },
        tokensUsed
      });

      return {
        message: response,
        context: {
          selectedStocks: request.selectedStocks,
          financialData: financialContext
        },
        tokensUsed
      };
    } catch (error) {
      logger.error('Error generating AI response:', error);
      throw new Error('Failed to generate AI response');
    }
  }

  async analyzeStocks(request: AnalysisRequest): Promise<any> {
    try {
      // Get financial data for all requested stocks
      const stocksData = await Promise.all(
        request.symbols.map(async (symbol) => {
          const [overview, ratios, quote] = await Promise.all([
            financialDataService.getFinancialData(symbol, 'overview', '1Y'),
            financialDataService.getFinancialRatios(symbol, 3),
            financialDataService.getQuote(symbol)
          ]);

          return { symbol, overview, ratios, quote };
        })
      );

      // Build analysis prompt based on type
      let analysisPrompt = '';
      switch (request.analysisType) {
        case 'overview':
          analysisPrompt = this.buildOverviewPrompt(stocksData);
          break;
        case 'detailed':
          analysisPrompt = this.buildDetailedPrompt(stocksData);
          break;
        case 'comparison':
          analysisPrompt = this.buildComparisonPrompt(stocksData);
          break;
      }

      const completion = await this.openai.chat.completions.create({
        model: this.model,
        messages: [
          {
            role: 'system',
            content: 'You are a professional financial analyst. Provide comprehensive, data-driven analysis based on the financial information provided. Use specific numbers and ratios to support your analysis.'
          },
          {
            role: 'user',
            content: analysisPrompt
          }
        ],
        max_tokens: 1500,
        temperature: 0.3
      });

      const analysis = completion.choices[0]?.message?.content || 'Analysis could not be generated.';

      // Save analysis to chat history
      await this.saveChatHistory({
        userId: request.userId,
        message: `Stock analysis request: ${request.symbols.join(', ')} (${request.analysisType})`,
        response: analysis,
        context: {
          analysisType: request.analysisType,
          symbols: request.symbols,
          stocksData
        },
        tokensUsed: completion.usage?.total_tokens || 0
      });

      return {
        analysis,
        symbols: request.symbols,
        analysisType: request.analysisType,
        stocksData
      };
    } catch (error) {
      logger.error('Error analyzing stocks:', error);
      throw new Error('Failed to analyze stocks');
    }
  }

  async getChatHistory(userId: string, limit: number = 50, offset: number = 0): Promise<any[]> {
    try {
      const history = await prisma.chatHistory.findMany({
        where: { userId },
        orderBy: { createdAt: 'desc' },
        take: limit,
        skip: offset,
        select: {
          id: true,
          message: true,
          response: true,
          createdAt: true,
          tokensUsed: true
        }
      });

      return history.reverse(); // Return in chronological order
    } catch (error) {
      logger.error('Error fetching chat history:', error);
      return [];
    }
  }

  async clearChatHistory(userId: string): Promise<void> {
    try {
      await prisma.chatHistory.deleteMany({
        where: { userId }
      });
    } catch (error) {
      logger.error('Error clearing chat history:', error);
      throw new Error('Failed to clear chat history');
    }
  }

  private async getFinancialContext(symbols: string[]): Promise<any> {
    if (symbols.length === 0) return {};

    try {
      const context: any = {};
      
      for (const symbol of symbols) {
        const [overview, ratios, quote] = await Promise.all([
          financialDataService.getFinancialData(symbol, 'overview', '1Y'),
          financialDataService.getFinancialRatios(symbol, 2),
          financialDataService.getQuote(symbol)
        ]);

        context[symbol] = {
          overview,
          ratios,
          quote,
          lastUpdated: new Date().toISOString()
        };
      }

      return context;
    } catch (error) {
      logger.error('Error getting financial context:', error);
      return {};
    }
  }

  private buildSystemPrompt(financialContext: any): string {
    let prompt = `You are FinAdvisor AI, a professional financial analyst assistant. You provide accurate, data-driven financial analysis and investment insights.

Key capabilities:
- Analyze financial statements, ratios, and market data
- Provide investment recommendations based on fundamental analysis
- Explain complex financial concepts in clear terms
- Compare stocks and identify trends
- Answer questions about market conditions and economic factors

Guidelines:
- Always base responses on factual financial data when available
- Clearly state when information is limited or when making assumptions
- Provide balanced analysis including both risks and opportunities
- Use specific numbers and ratios to support your analysis
- Avoid giving direct financial advice; instead, provide analytical insights

`;

    if (Object.keys(financialContext).length > 0) {
      prompt += `\nCurrent Financial Data Available:\n`;
      Object.keys(financialContext).forEach(symbol => {
        const data = financialContext[symbol];
        prompt += `\n${symbol}:`;
        if (data.quote) {
          prompt += `\n- Current Price: $${data.quote.price} (${data.quote.changePercent}%)`;
        }
        if (data.overview) {
          prompt += `\n- P/E Ratio: ${data.overview.PERatio || 'N/A'}`;
          prompt += `\n- Market Cap: ${data.overview.MarketCapitalization || 'N/A'}`;
          prompt += `\n- Sector: ${data.overview.Sector || 'N/A'}`;
        }
        if (data.ratios && data.ratios.length > 0) {
          const latestRatio = data.ratios[0];
          prompt += `\n- ROE: ${latestRatio.returnOnEquity?.toFixed(2)}%`;
          prompt += `\n- Debt/Equity: ${latestRatio.debtToEquity?.toFixed(2)}`;
        }
      });
    }

    return prompt;
  }

  private async getRecentChatHistory(userId: string, limit: number): Promise<any[]> {
    try {
      return await prisma.chatHistory.findMany({
        where: { userId },
        orderBy: { createdAt: 'desc' },
        take: limit,
        select: {
          message: true,
          response: true
        }
      });
    } catch (error) {
      logger.error('Error getting recent chat history:', error);
      return [];
    }
  }

  private async saveChatHistory(data: {
    userId: string;
    message: string;
    response: string;
    context: any;
    tokensUsed: number;
  }): Promise<void> {
    try {
      await prisma.chatHistory.create({
        data: {
          userId: data.userId,
          message: data.message,
          response: data.response,
          context: data.context,
          tokensUsed: data.tokensUsed
        }
      });
    } catch (error) {
      logger.error('Error saving chat history:', error);
    }
  }

  private buildOverviewPrompt(stocksData: any[]): string {
    let prompt = `Provide a brief overview analysis for the following stocks:\n\n`;
    
    stocksData.forEach(stock => {
      prompt += `${stock.symbol}:\n`;
      prompt += `- Current Price: $${stock.quote.price} (${stock.quote.changePercent >= 0 ? '+' : ''}${stock.quote.changePercent}%)\n`;
      if (stock.overview.PERatio) prompt += `- P/E Ratio: ${stock.overview.PERatio}\n`;
      if (stock.overview.Sector) prompt += `- Sector: ${stock.overview.Sector}\n`;
      if (stock.ratios.length > 0) {
        const latest = stock.ratios[0];
        if (latest.returnOnEquity) prompt += `- ROE: ${latest.returnOnEquity.toFixed(2)}%\n`;
      }
      prompt += `\n`;
    });

    prompt += `Please provide a concise overview of each stock's current position, key strengths, and any notable concerns.`;
    
    return prompt;
  }

  private buildDetailedPrompt(stocksData: any[]): string {
    let prompt = `Provide a detailed fundamental analysis for the following stocks:\n\n`;
    
    stocksData.forEach(stock => {
      prompt += `${stock.symbol} - ${stock.overview.Name || 'N/A'}:\n`;
      prompt += `Current Metrics:\n`;
      prompt += `- Price: $${stock.quote.price} (${stock.quote.changePercent >= 0 ? '+' : ''}${stock.quote.changePercent}%)\n`;
      prompt += `- Market Cap: ${stock.overview.MarketCapitalization || 'N/A'}\n`;
      prompt += `- P/E Ratio: ${stock.overview.PERatio || 'N/A'}\n`;
      prompt += `- EPS: ${stock.overview.EPS || 'N/A'}\n`;
      prompt += `- Dividend Yield: ${stock.overview.DividendYield || 'N/A'}\n`;
      
      if (stock.ratios.length > 0) {
        prompt += `Financial Ratios (Recent Years):\n`;
        stock.ratios.forEach((ratio: any) => {
          prompt += `  ${ratio.year}: ROE: ${ratio.returnOnEquity?.toFixed(2)}%, ROA: ${ratio.returnOnAssets?.toFixed(2)}%, Debt/Equity: ${ratio.debtToEquity?.toFixed(2)}\n`;
        });
      }
      prompt += `\n`;
    });

    prompt += `Please provide a comprehensive analysis including valuation, financial health, growth prospects, and investment thesis for each stock.`;
    
    return prompt;
  }

  private buildComparisonPrompt(stocksData: any[]): string {
    let prompt = `Compare the following stocks across key financial metrics:\n\n`;
    
    stocksData.forEach(stock => {
      prompt += `${stock.symbol}: Price $${stock.quote.price}, P/E ${stock.overview.PERatio || 'N/A'}, Market Cap ${stock.overview.MarketCapitalization || 'N/A'}`;
      if (stock.ratios.length > 0) {
        const latest = stock.ratios[0];
        prompt += `, ROE ${latest.returnOnEquity?.toFixed(2)}%, Debt/Equity ${latest.debtToEquity?.toFixed(2)}`;
      }
      prompt += `\n`;
    });

    prompt += `\nPlease provide a comparative analysis highlighting:\n1. Valuation comparison\n2. Financial strength comparison\n3. Growth prospects\n4. Risk assessment\n5. Investment recommendation ranking`;
    
    return prompt;
  }
}

export const aiChatService = new AIChatService();