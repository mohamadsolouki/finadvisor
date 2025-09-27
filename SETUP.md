# Development Environment Setup

To get your FinAdvisor platform up and running, follow these steps:

## Prerequisites

1. **Node.js 18+** - [Download here](https://nodejs.org/)
2. **PostgreSQL** - [Download here](https://www.postgresql.org/download/)
3. **Git** - [Download here](https://git-scm.com/)

## API Keys Required

You'll need to sign up for these services to get API keys:

1. **OpenAI API Key** - [Get it here](https://platform.openai.com/api-keys)
2. **Alpha Vantage API Key** - [Get it here](https://www.alphavantage.co/support/#api-key) (Free tier available)

## Quick Start

1. **Clone and install dependencies:**
```bash
cd finadvisor
npm run install:all
```

2. **Set up your database:**
Create a PostgreSQL database named `finadvisor` and update the connection string.

3. **Configure environment variables:**

**Backend (.env):**
```bash
cd backend
cp .env.example .env
```

Edit the `.env` file with your actual values:
```
NODE_ENV=development
PORT=3001
DATABASE_URL="postgresql://username:password@localhost:5432/finadvisor"
JWT_SECRET="your-super-secret-jwt-key-change-this-in-production"
JWT_EXPIRES_IN="7d"
OPENAI_API_KEY="your-openai-api-key"
ALPHA_VANTAGE_API_KEY="your-alpha-vantage-api-key"
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100
FRONTEND_URL="http://localhost:3000"
```

**Frontend (.env.local):**
```bash
cd ../frontend
cp .env.local.example .env.local
```

Edit the `.env.local` file:
```
NEXT_PUBLIC_API_URL=http://localhost:3001/api
```

4. **Initialize the database:**
```bash
cd backend
npx prisma migrate dev
```

Note: No seed data is used. Stock data will be fetched from Alpha Vantage API when users search for stocks.

5. **Start the development servers:**
```bash
# From the root directory
npm run dev
```

This will start:
- Backend API on `http://localhost:3001`
- Frontend on `http://localhost:3000`

## Project Structure Overview

```
finadvisor/
├── backend/                 # Express.js API server
│   ├── src/
│   │   ├── routes/         # API endpoints
│   │   ├── services/       # Business logic
│   │   ├── middleware/     # Authentication, error handling
│   │   └── utils/          # Logging, helpers
│   ├── prisma/             # Database schema and migrations
│   └── package.json
├── frontend/               # Next.js React application
│   ├── src/
│   │   ├── app/           # App Router pages
│   │   ├── components/    # Reusable UI components
│   │   ├── stores/        # State management
│   │   └── lib/           # Utilities and API clients
│   └── package.json
├── shared/                 # Shared types and utilities
└── package.json           # Root package.json for monorepo
```

## Key Features Implemented

### ✅ Backend Infrastructure
- Express.js REST API with TypeScript
- PostgreSQL database with Prisma ORM
- JWT authentication
- Rate limiting and security middleware
- Comprehensive logging with Winston
- Error handling and validation

### ✅ Database Schema
- Users and authentication
- Stock data and user selections (populated from Alpha Vantage API)
- Financial ratios and historical data
- AI chat history
- No seed data - all data fetched from reliable financial APIs

### ✅ API Endpoints
- **Authentication:** `/api/auth/login`, `/api/auth/register`
- **Stocks:** `/api/stocks`, `/api/stocks/search`, `/api/stocks/select`
- **Financial Data:** `/api/financial-data/:symbol`, `/api/financial-data/:symbol/ratios`
- **AI Chat:** `/api/chat`, `/api/chat/analyze`
- **User Management:** `/api/users/profile`

### ✅ Financial Data Integration
- Alpha Vantage API integration for real-time data
- No mock data - requires valid API key for operation
- Caching system for API responses
- Support for quotes, historical data, ratios, and earnings
- Stock search and discovery through Alpha Vantage Symbol Search

### ✅ AI Integration
- OpenAI GPT-4 integration
- Context-aware responses with financial data
- Stock analysis capabilities
- Chat history management

## Next Steps

After setup, you can:

1. **Register a new account** 
2. **Search for stocks** using company names or symbols via Alpha Vantage API
3. **Select stocks** to add to your portfolio
4. **View real-time financial data** and ratios for selected stocks
5. **Chat with the AI** for financial analysis and insights using live data
6. **Explore the data visualization** features with accurate market data

## Development Guidelines

- The backend API is fully functional with mock data
- Frontend components can be built to consume the API
- All API endpoints include proper error handling and validation
- The system is designed to handle real API keys when available
- Database migrations are version controlled

## Troubleshooting

**Database Connection Issues:**
- Ensure PostgreSQL is running
- Check the DATABASE_URL format
- Verify database exists and user has proper permissions

**API Key Issues:**
- Verify OpenAI API key has sufficient credits
- Check Alpha Vantage API key is active and has sufficient quota
- **CRITICAL**: Valid API keys are required - no fallback data available
- Free Alpha Vantage tier allows 5 API requests per minute, 500 per day

**Port Conflicts:**
- Backend runs on port 3001
- Frontend runs on port 3000
- Change ports in respective package.json files if needed

**Stock Search Issues:**
- If no results appear, verify Alpha Vantage API key is working
- Search uses Alpha Vantage Symbol Search - requires internet connection
- Rate limits may cause temporary failures - wait and retry

## Architecture Highlights

This is a **production-ready foundation** with:
- **Scalable monorepo structure**
- **Type-safe development** with TypeScript
- **Modern React** with Next.js 14 and App Router
- **Professional API design** with proper REST conventions
- **Security best practices** built-in
- **Comprehensive error handling**
- **Real-time financial data** integration ready
- **AI-powered analysis** capabilities