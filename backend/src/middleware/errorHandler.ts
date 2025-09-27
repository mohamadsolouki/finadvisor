import { Request, Response, NextFunction } from 'express';
import { logger } from '../utils/logger';

export interface AppError extends Error {
  statusCode?: number;
  isOperational?: boolean;
}

export const errorHandler = (
  err: AppError,
  req: Request,
  res: Response,
  next: NextFunction
) => {
  err.statusCode = err.statusCode || 500;
  err.isOperational = err.isOperational || false;

  logger.error(`${err.statusCode} - ${err.message} - ${req.originalUrl} - ${req.method} - ${req.ip}`);

  if (process.env.NODE_ENV === 'development') {
    res.status(err.statusCode).json({
      error: err.message,
      stack: err.stack,
      statusCode: err.statusCode
    });
  } else {
    // Production error response
    if (err.isOperational) {
      res.status(err.statusCode).json({
        error: err.message
      });
    } else {
      // Programming or other unknown error: don't leak error details
      res.status(500).json({
        error: 'Something went wrong!'
      });
    }
  }
};