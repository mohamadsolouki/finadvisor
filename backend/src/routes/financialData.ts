import express from 'express';
import { authenticateToken, AuthenticatedRequest } from '../middleware/auth';
import { financialDataService } from '../services/financialDataService';
import { logger } from '../utils/logger';

const router = express.Router();

// Get financial data for a specific stock
router.get('/:symbol', authenticateToken, async (req: AuthenticatedRequest, res, next) => {
  try {
    const { symbol } = req.params;
    const { period = '1Y', dataType = 'overview' } = req.query;

    const data = await financialDataService.getFinancialData(
      symbol as string,
      dataType as string,
      period as string
    );

    res.json({
      symbol: symbol.toUpperCase(),
      data,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error(`Error fetching financial data for ${req.params.symbol}:`, error);
    next(error);
  }
});

// Get financial ratios for a stock
router.get('/:symbol/ratios', authenticateToken, async (req: AuthenticatedRequest, res, next) => {
  try {
    const { symbol } = req.params;
    const { years = 5 } = req.query;

    const ratios = await financialDataService.getFinancialRatios(
      symbol as string,
      parseInt(years as string)
    );

    res.json({
      symbol: symbol.toUpperCase(),
      ratios,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error(`Error fetching ratios for ${req.params.symbol}:`, error);
    next(error);
  }
});

// Get historical data for a stock
router.get('/:symbol/historical', authenticateToken, async (req: AuthenticatedRequest, res, next) => {
  try {
    const { symbol } = req.params;
    const { period = '1Y', interval = 'daily' } = req.query;

    const historicalData = await financialDataService.getHistoricalData(
      symbol as string,
      period as string,
      interval as string
    );

    res.json({
      symbol: symbol.toUpperCase(),
      historicalData,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error(`Error fetching historical data for ${req.params.symbol}:`, error);
    next(error);
  }
});

// Get real-time quote for a stock
router.get('/:symbol/quote', authenticateToken, async (req: AuthenticatedRequest, res, next) => {
  try {
    const { symbol } = req.params;

    const quote = await financialDataService.getQuote(symbol as string);

    res.json({
      symbol: symbol.toUpperCase(),
      quote,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error(`Error fetching quote for ${req.params.symbol}:`, error);
    next(error);
  }
});

// Get earnings data for a stock
router.get('/:symbol/earnings', authenticateToken, async (req: AuthenticatedRequest, res, next) => {
  try {
    const { symbol } = req.params;

    const earnings = await financialDataService.getEarnings(symbol as string);

    res.json({
      symbol: symbol.toUpperCase(),
      earnings,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error(`Error fetching earnings for ${req.params.symbol}:`, error);
    next(error);
  }
});

export default router;