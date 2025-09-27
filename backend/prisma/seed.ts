import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  console.log('🗄️ Database initialization...');
  
  // Initialize database schema only - no sample data
  // Stock data will be populated from financial APIs when users search/select stocks
  
  console.log('✅ Database schema initialized successfully');
  console.log('📈 Stock data will be fetched from Alpha Vantage API when needed');
  console.log('🔑 Ensure ALPHA_VANTAGE_API_KEY and OPENAI_API_KEY are configured');
}

main()
  .catch((e) => {
    console.error('❌ Error seeding database:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });