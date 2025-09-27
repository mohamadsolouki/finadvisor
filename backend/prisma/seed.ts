import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  console.log('🌱 Starting database seeding...');

  // Create sample stocks
  const stocks = [
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      exchange: 'NASDAQ',
      sector: 'Technology',
      industry: 'Consumer Electronics',
      marketCap: BigInt(3000000000000) // 3T
    },
    {
      symbol: 'MSFT',
      name: 'Microsoft Corporation',
      exchange: 'NASDAQ',
      sector: 'Technology',
      industry: 'Software',
      marketCap: BigInt(2800000000000) // 2.8T
    },
    {
      symbol: 'GOOGL',
      name: 'Alphabet Inc.',
      exchange: 'NASDAQ',
      sector: 'Technology',
      industry: 'Internet Services',
      marketCap: BigInt(1700000000000) // 1.7T
    },
    {
      symbol: 'AMZN',
      name: 'Amazon.com Inc.',
      exchange: 'NASDAQ',
      sector: 'Consumer Discretionary',
      industry: 'E-commerce',
      marketCap: BigInt(1500000000000) // 1.5T
    },
    {
      symbol: 'TSLA',
      name: 'Tesla Inc.',
      exchange: 'NASDAQ',
      sector: 'Consumer Discretionary',
      industry: 'Electric Vehicles',
      marketCap: BigInt(800000000000) // 800B
    },
    {
      symbol: 'NVDA',
      name: 'NVIDIA Corporation',
      exchange: 'NASDAQ',
      sector: 'Technology',
      industry: 'Semiconductors',
      marketCap: BigInt(1200000000000) // 1.2T
    },
    {
      symbol: 'JPM',
      name: 'JPMorgan Chase & Co.',
      exchange: 'NYSE',
      sector: 'Financial Services',
      industry: 'Banking',
      marketCap: BigInt(500000000000) // 500B
    },
    {
      symbol: 'JNJ',
      name: 'Johnson & Johnson',
      exchange: 'NYSE',
      sector: 'Healthcare',
      industry: 'Pharmaceuticals',
      marketCap: BigInt(450000000000) // 450B
    },
    {
      symbol: 'V',
      name: 'Visa Inc.',
      exchange: 'NYSE',
      sector: 'Financial Services',
      industry: 'Payment Processing',
      marketCap: BigInt(500000000000) // 500B
    },
    {
      symbol: 'PG',
      name: 'Procter & Gamble Co.',
      exchange: 'NYSE',
      sector: 'Consumer Staples',
      industry: 'Personal Care Products',
      marketCap: BigInt(380000000000) // 380B
    }
  ];

  for (const stock of stocks) {
    await prisma.stock.upsert({
      where: { symbol: stock.symbol },
      update: {},
      create: stock
    });
  }

  console.log(`✅ Created ${stocks.length} stocks`);

  // Create sample financial ratios for Apple (AAPL)
  const appleStock = await prisma.stock.findUnique({
    where: { symbol: 'AAPL' }
  });

  if (appleStock) {
    const ratiosData = [
      {
        year: 2023,
        peRatio: 28.5,
        pbRatio: 40.2,
        debtToEquity: 1.73,
        currentRatio: 1.01,
        quickRatio: 0.96,
        returnOnEquity: 147.4,
        returnOnAssets: 27.6,
        grossMargin: 44.1,
        operatingMargin: 29.8,
        netMargin: 25.3,
        dividendYield: 0.5
      },
      {
        year: 2022,
        peRatio: 23.2,
        pbRatio: 35.8,
        debtToEquity: 1.95,
        currentRatio: 0.95,
        quickRatio: 0.89,
        returnOnEquity: 175.1,
        returnOnAssets: 28.0,
        grossMargin: 43.3,
        operatingMargin: 30.3,
        netMargin: 25.7,
        dividendYield: 0.6
      },
      {
        year: 2021,
        peRatio: 30.9,
        pbRatio: 39.4,
        debtToEquity: 1.73,
        currentRatio: 1.07,
        quickRatio: 1.03,
        returnOnEquity: 147.4,
        returnOnAssets: 26.3,
        grossMargin: 41.8,
        operatingMargin: 29.8,
        netMargin: 25.9,
        dividendYield: 0.7
      }
    ];

    for (const ratioData of ratiosData) {
      await prisma.financialRatio.upsert({
        where: {
          stockId_year: {
            stockId: appleStock.id,
            year: ratioData.year
          }
        },
        update: {},
        create: {
          stockId: appleStock.id,
          ...ratioData
        }
      });
    }

    console.log(`✅ Created financial ratios for AAPL`);
  }

  console.log('🎉 Database seeding completed!');
}

main()
  .catch((e) => {
    console.error('❌ Error seeding database:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });