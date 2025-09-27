import express from 'express';
import { PrismaClient } from '@prisma/client';
import { authenticateToken, AuthenticatedRequest } from '../middleware/auth';
import { body, validationResult } from 'express-validator';
import { logger } from '../utils/logger';
import { stockDataService } from '../services/stockDataService';

const router = express.Router();
const prisma = new PrismaClient();

// Get all available stocks
router.get('/', async (req, res, next) => {
  try {
    // Fetch stocks from database (populated from API calls)
    const stocks = await prisma.stock.findMany({
      select: {
        id: true,
        symbol: true,
        name: true,
        exchange: true,
        sector: true,
        industry: true,
        marketCap: true,
        isActive: true
      },
      where: { isActive: true },
      orderBy: { symbol: 'asc' }
    });

    if (stocks.length === 0) {
      return res.json({
        stocks: [],
        total: 0,
        message: 'No stocks available. Search for specific symbols to populate data from financial APIs.'
      });
    }

    res.json({
      stocks,
      total: stocks.length
    });
  } catch (error) {
    logger.error('Error fetching stocks:', error);
    next(error);
  }
});

// Search stocks using Alpha Vantage API
router.get('/search', async (req, res, next) => {
  try {
    const { q, limit = 10 } = req.query;
    
    if (!q || typeof q !== 'string' || q.trim().length === 0) {
      return res.status(400).json({
        error: 'Search query (q) is required and must be a non-empty string'
      });
    }

    // Search for stocks using Alpha Vantage API
    const apiResults = await stockDataService.searchStocks(q as string);
    
    // Limit results
    const limitedResults = apiResults.slice(0, parseInt(limit as string));
    
    // For each result, create or update in database if it doesn't exist
    const stocksWithDbInfo = await Promise.all(
      limitedResults.map(async (result) => {
        try {
          // Try to get stock from database first
          let stock = await prisma.stock.findUnique({
            where: { symbol: result.symbol }
          });

          // If not in database, create it
          if (!stock) {
            stock = await stockDataService.createOrUpdateStock(result.symbol);
          }

          return {
            id: stock.id,
            symbol: result.symbol,
            name: result.name,
            exchange: result.region,
            sector: stock.sector,
            industry: stock.industry,
            marketCap: stock.marketCap?.toString(),
            type: result.type,
            matchScore: parseFloat(result.matchScore)
          };
        } catch (error) {
          logger.warn(`Could not process search result for ${result.symbol}:`, error);
          return {
            symbol: result.symbol,
            name: result.name,
            exchange: result.region,
            type: result.type,
            matchScore: parseFloat(result.matchScore),
            error: 'Could not fetch detailed information'
          };
        }
      })
    );

    res.json({
      stocks: stocksWithDbInfo,
      total: stocksWithDbInfo.length,
      query: q
    });
  } catch (error) {
    logger.error('Error searching stocks:', error);
    next(error);
  }
});

// Get user's selected stocks
router.get('/selected', authenticateToken, async (req: AuthenticatedRequest, res, next) => {
  try {
    const userStocks = await prisma.userStock.findMany({
      where: { userId: req.user!.id },
      include: {
        stock: {
          select: {
            id: true,
            symbol: true,
            name: true,
            exchange: true,
            sector: true,
            industry: true,
            marketCap: true
          }
        }
      },
      orderBy: { createdAt: 'desc' }
    });

    res.json({
      selectedStocks: userStocks.map(us => ({
        ...us.stock,
        selectedAt: us.createdAt,
        notes: us.notes
      }))
    });
  } catch (error) {
    logger.error('Error fetching selected stocks:', error);
    next(error);
  }
});

// Select/add stock to user's portfolio by symbol
router.post('/select', [
  authenticateToken,
  body('symbol').isString().isLength({ min: 1, max: 10 }).withMessage('Valid stock symbol is required'),
  body('notes').optional().isString().isLength({ max: 500 })
], async (req: AuthenticatedRequest, res, next) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { symbol, notes } = req.body;

    // Create or get stock from API
    const stock = await stockDataService.createOrUpdateStock(symbol);

    // Check if already selected
    const existingSelection = await prisma.userStock.findUnique({
      where: {
        userId_stockId: {
          userId: req.user!.id,
          stockId: stock.id
        }
      }
    });

    if (existingSelection) {
      return res.status(409).json({ error: 'Stock already selected' });
    }

    // Add to user's selection
    const userStock = await prisma.userStock.create({
      data: {
        userId: req.user!.id,
        stockId: stock.id,
        notes: notes || null
      },
      include: {
        stock: {
          select: {
            id: true,
            symbol: true,
            name: true,
            exchange: true,
            sector: true,
            industry: true,
            marketCap: true
          }
        }
      }
    });

    res.status(201).json({
      message: 'Stock selected successfully',
      selectedStock: {
        ...userStock.stock,
        selectedAt: userStock.createdAt,
        notes: userStock.notes
      }
    });
  } catch (error) {
    logger.error('Error selecting stock:', error);
    next(error);
  }
});

// Remove stock from user's selection
router.delete('/select/:stockId', authenticateToken, async (req: AuthenticatedRequest, res, next) => {
  try {
    const { stockId } = req.params;

    const deletedUserStock = await prisma.userStock.deleteMany({
      where: {
        userId: req.user!.id,
        stockId
      }
    });

    if (deletedUserStock.count === 0) {
      return res.status(404).json({ error: 'Stock selection not found' });
    }

    res.json({ message: 'Stock removed from selection' });
  } catch (error) {
    logger.error('Error removing stock selection:', error);
    next(error);
  }
});

export default router;