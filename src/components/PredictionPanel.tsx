import { TechnicalIndicators } from "../types";

interface PredictionPanelProps {
  indicators: TechnicalIndicators;
}

export function PredictionPanel({ indicators }: PredictionPanelProps) {
  const getRsiStatus = (rsi: number) => {
    if (rsi > 70) return { label: "超买", color: "text-red-600 bg-red-50" };
    if (rsi < 30) return { label: "超卖", color: "text-green-600 bg-green-50" };
    return { label: "正常", color: "text-blue-600 bg-blue-50" };
  };

  const rsiStatus = getRsiStatus(indicators.rsi);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6">
      <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">
        📊 技术指标分析
      </h3>

      <div className="space-y-4">
        {/* 移动平均线 */}
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 text-center">
            <div className="text-sm text-gray-500 dark:text-gray-400">MA5</div>
            <div className="text-lg font-semibold text-gray-800 dark:text-white">
              {indicators.ma5.toFixed(2)}
            </div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 text-center">
            <div className="text-sm text-gray-500 dark:text-gray-400">MA10</div>
            <div className="text-lg font-semibold text-gray-800 dark:text-white">
              {indicators.ma10.toFixed(2)}
            </div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 text-center">
            <div className="text-sm text-gray-500 dark:text-gray-400">MA20</div>
            <div className="text-lg font-semibold text-gray-800 dark:text-white">
              {indicators.ma20.toFixed(2)}
            </div>
          </div>
        </div>

        {/* RSI 指标 */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-gray-500 dark:text-gray-400">
              RSI (相对强弱指数)
            </span>
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${rsiStatus.color}`}>
              {rsiStatus.label}
            </span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="flex-1 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${indicators.rsi}%` }}
              />
            </div>
            <span className="text-lg font-semibold text-gray-800 dark:text-white">
              {indicators.rsi.toFixed(2)}
            </span>
          </div>
        </div>

        {/* MACD */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
          <div className="text-sm text-gray-500 dark:text-gray-400 mb-2">
            MACD (指数平滑异同移动平均线)
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-lg font-semibold text-gray-800 dark:text-white">
              {indicators.macd.toFixed(4)}
            </span>
            <span className={`text-sm ${indicators.macd >= 0 ? "text-green-600" : "text-red-600"}`}>
              {indicators.macd >= 0 ? "↑" : "↓"}
            </span>
          </div>
        </div>

        {/* 交易信号 */}
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
          <h4 className="text-sm font-medium text-blue-800 dark:text-blue-300 mb-2">
            💡 交易信号提示
          </h4>
          <ul className="text-sm text-blue-700 dark:text-blue-400 space-y-1">
            {indicators.ma5 > indicators.ma10 && (
              <li>• MA5 上穿 MA10，短期看涨信号</li>
            )}
            {indicators.ma5 < indicators.ma10 && (
              <li>• MA5 下穿 MA10，短期看跌信号</li>
            )}
            {indicators.rsi > 70 && (
              <li>• RSI 超买区间，注意回调风险</li>
            )}
            {indicators.rsi < 30 && (
              <li>• RSI 超卖区间，可能存在反弹机会</li>
            )}
            {indicators.macd > 0 && (
              <li>• MACD 为正，多头趋势</li>
            )}
            {indicators.macd < 0 && (
              <li>• MACD 为负，空头趋势</li>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
}
