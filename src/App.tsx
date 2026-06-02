import { useState, useCallback, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { StockChart } from "./components/StockChart";
import { PredictionPanel } from "./components/PredictionPanel";
import { AnalysisPanel } from "./components/AnalysisPanel";
import { Header } from "./components/Header";
import { StockData, AnalysisResult, TechnicalIndicators, ModelInfo } from "./types";

function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stockData, setStockData] = useState<StockData[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [indicators, setIndicators] = useState<TechnicalIndicators | null>(null);
  const [symbol, setSymbol] = useState("AAPL");
  const [startDate, setStartDate] = useState("2023-01-01");
  const [endDate, setEndDate] = useState(new Date().toISOString().split("T")[0]);
  const [dataSource, setDataSource] = useState<"yahoo" | "stooq">("yahoo");
  
  // API 配置状态
  const [apiKey, setApiKey] = useState("");
  const [apiBase, setApiBase] = useState("https://api.deepseek.com");
  const [showApiConfig, setShowApiConfig] = useState(false);
  
  // 模型选择状态
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState("deepseek-chat");
  const [loadingModels, setLoadingModels] = useState(false);

  // 获取模型列表
  const fetchModels = useCallback(async () => {
    if (!apiKey) return;
    
    setLoadingModels(true);
    try {
      const modelList = await invoke<ModelInfo[]>("fetch_models", {
        apiKey,
        apiBase,
      });
      setModels(modelList);
      if (modelList.length > 0 && !modelList.find(m => m.id === selectedModel)) {
        setSelectedModel(modelList[0].id);
      }
    } catch (err) {
      console.error("Failed to fetch models:", err);
      // 使用默认模型列表
      setModels([
        { id: "deepseek-chat", name: "DeepSeek Chat", owned_by: "deepseek" },
        { id: "deepseek-coder", name: "DeepSeek Coder", owned_by: "deepseek" },
      ]);
    } finally {
      setLoadingModels(false);
    }
  }, [apiKey, apiBase, selectedModel]);

  // API Key 或 Base 变化时获取模型列表
  useEffect(() => {
    const timer = setTimeout(() => {
      if (apiKey) {
        fetchModels();
      }
    }, 500);
    return () => clearTimeout(timer);
  }, [apiKey, apiBase, fetchModels]);

  const handleFetchData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // 获取股票数据
      const data = await invoke<StockData[]>("fetch_stock_data", {
        symbol,
        startDate,
        endDate,
        source: dataSource,
      });

      setStockData(data);

      // 计算技术指标
      const ind = await invoke<TechnicalIndicators>("calculate_indicators", {
        data,
      });
      setIndicators(ind);

      // 调用 AI 分析
      const analysisResult = await invoke<AnalysisResult>("analyze_with_ai", {
        data,
        apiKey,
        apiBase,
        model: selectedModel,
      });
      setAnalysis(analysisResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [symbol, startDate, endDate, dataSource, apiKey, apiBase, selectedModel]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        {/* API 配置区域 */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-800 dark:text-white">
              🔑 AI 分析配置
            </h2>
            <button
              onClick={() => setShowApiConfig(!showApiConfig)}
              className="text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 text-sm font-medium"
            >
              {showApiConfig ? "收起" : "展开配置"}
            </button>
          </div>
          
          {showApiConfig && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    API Base URL
                  </label>
                  <input
                    type="text"
                    value={apiBase}
                    onChange={(e) => setApiBase(e.target.value)}
                    placeholder="https://api.deepseek.com"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    DeepSeek、OpenAI 或其他兼容 API 的地址
                  </p>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    API Key
                  </label>
                  <div className="relative">
                    <input
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder="sk-..."
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    />
                    <button
                      onClick={fetchModels}
                      disabled={loadingModels || !apiKey}
                      className="absolute right-2 top-1/2 -translate-y-1/2 px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loadingModels ? "加载中..." : "刷新模型"}
                    </button>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    填入后自动拉取可用模型列表
                  </p>
                </div>
              </div>
              
              {/* 模型选择 */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  选择模型
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                >
                  {models.length > 0 ? (
                    models.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name} ({model.owned_by})
                      </option>
                    ))
                  ) : (
                    <>
                      <option value="deepseek-chat">DeepSeek Chat (deepseek)</option>
                      <option value="deepseek-coder">DeepSeek Coder (deepseek)</option>
                      <option value="gpt-4o">GPT-4o (openai)</option>
                      <option value="gpt-4o-mini">GPT-4o Mini (openai)</option>
                    </>
                  )}
                </select>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {models.length > 0 
                    ? `已加载 ${models.length} 个模型` 
                    : "填入 API Key 后自动加载模型列表"}
                </p>
              </div>
            </div>
          )}
          
          {!showApiConfig && (
            <div className="flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-400">
              <span>📡 {apiBase}</span>
              <span>•</span>
              <span>🤖 {selectedModel}</span>
              <span>•</span>
              <span>🔑 {apiKey ? "已配置" : "未配置"}</span>
            </div>
          )}
        </div>

        {/* 股票查询表单 */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4 text-gray-800 dark:text-white">
            📊 股票数据查询
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                股票代码
              </label>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                placeholder="例如: AAPL, 000001.ss"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                开始日期
              </label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                结束日期
              </label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                数据源
              </label>
              <select
                value={dataSource}
                onChange={(e) => setDataSource(e.target.value as "yahoo" | "stooq")}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              >
                <option value="yahoo">Yahoo Finance</option>
                <option value="stooq">Stooq</option>
              </select>
            </div>
          </div>
          
          <div className="mt-4">
            <button
              onClick={handleFetchData}
              disabled={loading}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:ring-4 focus:ring-blue-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
            >
              {loading ? (
                <span className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  分析中...
                </span>
              ) : (
                "🚀 开始预测"
              )}
            </button>
          </div>
        </div>

        {/* 错误提示 */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-6">
            <p className="text-red-800 dark:text-red-200">{error}</p>
          </div>
        )}

        {/* 图表区域 */}
        {stockData.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <StockChart data={stockData} title="📈 历史价格走势" />
            {indicators && <PredictionPanel indicators={indicators} />}
          </div>
        )}

        {/* AI 分析区域 */}
        {analysis && (
          <AnalysisPanel analysis={analysis} />
        )}
      </main>
    </div>
  );
}

export default App;
