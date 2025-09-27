export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">FinAdvisor Dashboard</h1>
          <p className="text-gray-600 mt-2">AI-Powered Financial Analysis Platform</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Stock Selection</h2>
            <p className="text-gray-600">Choose stocks to analyze from our comprehensive database</p>
            <a href="/stocks" className="mt-4 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 inline-block text-center">
              Browse Stocks
            </a>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Financial Data</h2>
            <p className="text-gray-600">Real-time financial data and analysis</p>
            <a href="/financial-data" className="mt-4 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 inline-block text-center">
              View Data
            </a>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">AI Analysis</h2>
            <p className="text-gray-600">Get AI-powered insights and recommendations</p>
            <a href="/chat" className="mt-4 bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 inline-block text-center">
              Chat with AI
            </a>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Quick Stats</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">0</div>
              <div className="text-sm text-gray-600">Stocks Tracked</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">$0</div>
              <div className="text-sm text-gray-600">Portfolio Value</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-600">0</div>
              <div className="text-sm text-gray-600">AI Insights</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">Online</div>
              <div className="text-sm text-gray-600">Status</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}