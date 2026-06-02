// 股票数据类型
export interface StockData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// 预测结果类型
export interface PredictionResult {
  arima: number[];
  lstm: number;
}

// AI 分析结果类型
export interface AnalysisResult {
  summary: string;
  trend: string;
  indicators: TechnicalIndicators;
  prediction: string;
  advice: string;
  risk: string;
}

// 技术指标类型
export interface TechnicalIndicators {
  ma5: number;
  ma10: number;
  ma20: number;
  rsi: number;
  macd: number;
}

// 模型信息类型
export interface ModelInfo {
  id: string;
  name: string;
  owned_by: string;
}

// 数据源类型
export type DataSource = "yahoo" | "stooq" | "alpha_vantage";

// 应用状态类型
export interface AppState {
  loading: boolean;
  error: string | null;
  stockData: StockData[];
  predictions: PredictionResult | null;
  analysis: AnalysisResult | null;
}

// API 配置类型
export interface ApiConfig {
  deepseekApiKey: string;
  deepseekApiBase: string;
}
