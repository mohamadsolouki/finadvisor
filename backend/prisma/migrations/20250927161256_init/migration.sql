-- CreateEnum
CREATE TYPE "Role" AS ENUM ('USER', 'ADMIN');

-- CreateTable
CREATE TABLE "users" (
    "id" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "firstName" TEXT NOT NULL,
    "lastName" TEXT NOT NULL,
    "role" "Role" NOT NULL DEFAULT 'USER',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "stocks" (
    "id" TEXT NOT NULL,
    "symbol" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "exchange" TEXT NOT NULL,
    "sector" TEXT,
    "industry" TEXT,
    "marketCap" BIGINT,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "stocks_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "user_stocks" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "stockId" TEXT NOT NULL,
    "notes" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "user_stocks_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "financial_data" (
    "id" TEXT NOT NULL,
    "stockId" TEXT NOT NULL,
    "period" TEXT NOT NULL,
    "dataType" TEXT NOT NULL,
    "revenue" BIGINT,
    "netIncome" BIGINT,
    "totalAssets" BIGINT,
    "totalLiabilities" BIGINT,
    "shareholdersEquity" BIGINT,
    "operatingCashFlow" BIGINT,
    "freeCashFlow" BIGINT,
    "eps" DOUBLE PRECISION,
    "dividendPerShare" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "financial_data_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "financial_ratios" (
    "id" TEXT NOT NULL,
    "stockId" TEXT NOT NULL,
    "year" INTEGER NOT NULL,
    "peRatio" DOUBLE PRECISION,
    "pbRatio" DOUBLE PRECISION,
    "debtToEquity" DOUBLE PRECISION,
    "currentRatio" DOUBLE PRECISION,
    "quickRatio" DOUBLE PRECISION,
    "returnOnEquity" DOUBLE PRECISION,
    "returnOnAssets" DOUBLE PRECISION,
    "grossMargin" DOUBLE PRECISION,
    "operatingMargin" DOUBLE PRECISION,
    "netMargin" DOUBLE PRECISION,
    "assetTurnover" DOUBLE PRECISION,
    "inventoryTurnover" DOUBLE PRECISION,
    "receivablesTurnover" DOUBLE PRECISION,
    "dividendYield" DOUBLE PRECISION,
    "priceToSales" DOUBLE PRECISION,
    "priceToBook" DOUBLE PRECISION,
    "evToEbitda" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "financial_ratios_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "historical_data" (
    "id" TEXT NOT NULL,
    "stockId" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "open" DOUBLE PRECISION NOT NULL,
    "high" DOUBLE PRECISION NOT NULL,
    "low" DOUBLE PRECISION NOT NULL,
    "close" DOUBLE PRECISION NOT NULL,
    "adjustedClose" DOUBLE PRECISION,
    "volume" BIGINT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "historical_data_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "chat_history" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "message" TEXT NOT NULL,
    "response" TEXT NOT NULL,
    "context" JSONB,
    "tokensUsed" INTEGER,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "chat_history_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");

-- CreateIndex
CREATE UNIQUE INDEX "stocks_symbol_key" ON "stocks"("symbol");

-- CreateIndex
CREATE UNIQUE INDEX "user_stocks_userId_stockId_key" ON "user_stocks"("userId", "stockId");

-- CreateIndex
CREATE UNIQUE INDEX "financial_data_stockId_period_dataType_key" ON "financial_data"("stockId", "period", "dataType");

-- CreateIndex
CREATE UNIQUE INDEX "financial_ratios_stockId_year_key" ON "financial_ratios"("stockId", "year");

-- CreateIndex
CREATE UNIQUE INDEX "historical_data_stockId_date_key" ON "historical_data"("stockId", "date");

-- AddForeignKey
ALTER TABLE "user_stocks" ADD CONSTRAINT "user_stocks_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "user_stocks" ADD CONSTRAINT "user_stocks_stockId_fkey" FOREIGN KEY ("stockId") REFERENCES "stocks"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "financial_data" ADD CONSTRAINT "financial_data_stockId_fkey" FOREIGN KEY ("stockId") REFERENCES "stocks"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "financial_ratios" ADD CONSTRAINT "financial_ratios_stockId_fkey" FOREIGN KEY ("stockId") REFERENCES "stocks"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "historical_data" ADD CONSTRAINT "historical_data_stockId_fkey" FOREIGN KEY ("stockId") REFERENCES "stocks"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "chat_history" ADD CONSTRAINT "chat_history_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
