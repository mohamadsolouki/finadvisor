# FinAdvisor - AI Financial Analyst Platform

A comprehensive AI-powered financial analysis platform that provides intelligent stock analysis using real-time financial data and OpenAI integration.

## Features

- 📈 **Stock Selection & Analysis**: Choose from a comprehensive list of financial instruments
- 📊 **Financial Data Integration**: Real-time data from reliable financial sources
- 🤖 **AI-Powered Insights**: Context-aware financial analysis using OpenAI
- 📋 **Categorized Data Display**: Well-organized financial ratios and historical data
- 💬 **Interactive Chat**: Sidebar AI assistant with access to all financial data
- 📱 **Responsive Design**: Modern, user-friendly interface

## Tech Stack

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Chart.js** - Data visualization
- **Zustand** - State management

### Backend
- **Node.js** - Runtime
- **Express.js** - Web framework
- **TypeScript** - Type safety
- **Prisma** - Database ORM
- **PostgreSQL** - Database
- **OpenAI API** - AI integration
- **Alpha Vantage API** - Financial data

## Project Structure

```
finadvisor/
├── frontend/           # Next.js frontend application
├── backend/           # Express.js backend API
├── shared/           # Shared types and utilities
├── docs/             # Documentation
└── package.json      # Root package.json for monorepo
```

## Getting Started

### Prerequisites
- Node.js 18+
- PostgreSQL
- OpenAI API Key
- Financial Data API Key (Alpha Vantage recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mohamadsolouki/finadvisor.git
cd finadvisor
```

2. Install dependencies:
```bash
npm run install:all
```

3. Set up environment variables:
```bash
# Backend (.env)
DATABASE_URL="postgresql://username:password@localhost:5432/finadvisor"
OPENAI_API_KEY="your-openai-api-key"
ALPHA_VANTAGE_API_KEY="your-alpha-vantage-api-key"
JWT_SECRET="your-jwt-secret"
PORT=3001

# Frontend (.env.local)
NEXT_PUBLIC_API_URL="http://localhost:3001/api"
```

4. Set up the database:
```bash
cd backend
npx prisma migrate dev
npx prisma db seed
```

5. Start the development servers:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000` and the backend at `http://localhost:3001`.

## API Documentation

### Endpoints

- `GET /api/stocks` - Get list of available stocks
- `POST /api/stocks/select` - Select stocks for analysis
- `GET /api/financial-data/:symbol` - Get financial data for a stock
- `POST /api/chat` - AI chat endpoint
- `GET /api/ratios/:symbol` - Get financial ratios
- `GET /api/historical/:symbol` - Get historical data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.