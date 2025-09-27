import express from 'express';
import { authenticateToken, AuthenticatedRequest } from '../middleware/auth';
import { aiChatService } from '../services/aiChatService';
import { body, validationResult } from 'express-validator';
import { logger } from '../utils/logger';

const router = express.Router();

// AI Chat endpoint
router.post('/', [
  authenticateToken,
  body('message').trim().isLength({ min: 1, max: 2000 }).withMessage('Message must be between 1 and 2000 characters'),
  body('context').optional().isObject(),
  body('selectedStocks').optional().isArray()
], async (req: AuthenticatedRequest, res, next) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { message, context, selectedStocks } = req.body;

    const response = await aiChatService.generateResponse({
      userId: req.user!.id,
      message,
      context: context || {},
      selectedStocks: selectedStocks || []
    });

    res.json({
      response: response.message,
      context: response.context,
      timestamp: new Date().toISOString(),
      tokensUsed: response.tokensUsed
    });
  } catch (error) {
    logger.error('Error in AI chat:', error);
    next(error);
  }
});

// Get chat history
router.get('/history', authenticateToken, async (req: AuthenticatedRequest, res, next) => {
  try {
    const { limit = 50, offset = 0 } = req.query;

    const history = await aiChatService.getChatHistory(
      req.user!.id,
      parseInt(limit as string),
      parseInt(offset as string)
    );

    res.json({
      history,
      total: history.length
    });
  } catch (error) {
    logger.error('Error fetching chat history:', error);
    next(error);
  }
});

// Clear chat history
router.delete('/history', authenticateToken, async (req: AuthenticatedRequest, res, next) => {
  try {
    await aiChatService.clearChatHistory(req.user!.id);

    res.json({
      message: 'Chat history cleared successfully'
    });
  } catch (error) {
    logger.error('Error clearing chat history:', error);
    next(error);
  }
});

// Get AI analysis for specific stocks
router.post('/analyze', [
  authenticateToken,
  body('symbols').isArray({ min: 1, max: 10 }).withMessage('Must provide 1-10 stock symbols'),
  body('analysisType').optional().isIn(['overview', 'detailed', 'comparison'])
], async (req: AuthenticatedRequest, res, next) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { symbols, analysisType = 'overview' } = req.body;

    const analysis = await aiChatService.analyzeStocks({
      userId: req.user!.id,
      symbols,
      analysisType
    });

    res.json({
      analysis,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Error in stock analysis:', error);
    next(error);
  }
});

export default router;