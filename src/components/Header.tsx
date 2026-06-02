export function Header() {
  return (
    <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white text-xl">📈</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                Stock Predictor
              </h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                股票市场预测工具
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <span className="px-3 py-1 bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400 rounded-full text-xs font-medium">
              v1.0.0
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
