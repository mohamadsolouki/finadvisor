import jwt from 'jsonwebtoken';
import { Request, Response, NextFunction } from 'express';
import { AppError } from './errorHandler';

export interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    email: string;
    role: string;
  };
}

export const authenticateToken = (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    const error: AppError = new Error('Access token required');
    error.statusCode = 401;
    error.isOperational = true;
    return next(error);
  }

  jwt.verify(token, process.env.JWT_SECRET as string, (err: any, user: any) => {
    if (err) {
      const error: AppError = new Error('Invalid or expired token');
      error.statusCode = 403;
      error.isOperational = true;
      return next(error);
    }

    req.user = user;
    next();
  });
};

export const requireRole = (roles: string[]) => {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    if (!req.user) {
      const error: AppError = new Error('Authentication required');
      error.statusCode = 401;
      error.isOperational = true;
      return next(error);
    }

    if (!roles.includes(req.user.role)) {
      const error: AppError = new Error('Insufficient permissions');
      error.statusCode = 403;
      error.isOperational = true;
      return next(error);
    }

    next();
  };
};