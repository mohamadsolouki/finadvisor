import express from 'express';
import { PrismaClient } from '@prisma/client';
import { authenticateToken, AuthenticatedRequest } from '../middleware/auth';
import { body, validationResult } from 'express-validator';
import { logger } from '../utils/logger';

const router = express.Router();
const prisma = new PrismaClient();

// Get all available stocks
router.get('/', async (req, res, next) => {
  try {
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

    res.json({
      stocks,
      total: stocks.length
    });
  } catch (error) {
    logger.error('Error fetching stocks:', error);
    next(error);
  }
});

// Search stocks
router.get('/search', async (req, res, next) => {
  try {
    const { q, sector, exchange, limit = 50 } = req.query;
    
    const where: any = { isActive: true };
    
    if (q) {
      where.OR = [
        { symbol: { contains: q as string, mode: 'insensitive' } },
        { name: { contains: q as string, mode: 'insensitive' } }
      ];
    }
    
    if (sector) {
      where.sector = sector;
    }
    
    if (exchange) {
      where.exchange = exchange;
    }

    const stocks = await prisma.stock.findMany({
      where,
      take: parseInt(limit as string),
      select: {
        id: true,
        symbol: true,
        name: true,
        exchange: true,
        sector: true,
        industry: true,
        marketCap: true
      },
      orderBy: { symbol: 'asc' }
    });

    res.json({
      stocks,
      total: stocks.length
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

// Select/add stock to user's portfolio
router.post('/select', [
  authenticateToken,
  body('stockId').isUUID().withMessage('Valid stock ID is required'),
  body('notes').optional().isString().isLength({ max: 500 })
], async (req: AuthenticatedRequest, res, next) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { stockId, notes } = req.body;

    // Check if stock exists
    const stock = await prisma.stock.findUnique({
      where: { id: stockId }
    });

    if (!stock) {
      return res.status(404).json({ error: 'Stock not found' });
    }

    // Check if already selected
    const existingSelection = await prisma.userStock.findUnique({
      where: {
        userId_stockId: {
          userId: req.user!.id,
          stockId
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
        stockId,
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