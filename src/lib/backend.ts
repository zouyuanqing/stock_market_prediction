// Backend wrapper: uses the real Tauri (Rust) backend on desktop,
// and automatically falls back to an in-browser demo mode with mock data
// so the UI can be showcased on static hosting (e.g. GitHub Pages).
import { invoke } from "@tauri-apps/api/core";
import type { StockData, AnalysisResult } from "../types";

export const isTauri: boolean =
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

// Remember the symbol requested in demo mode so the AI analysis
// (which only receives data, not the symbol) can still label the report.
let lastDemoSymbol = "AAPL";

const S_DEEP_ANALYSIS = "深度分析报告";
const S_TREND = "趋势判断";
const S_TREND_BODY = "近 6 个月整体呈震荡上行趋势，MA5 已上穿 MA20 形成金叉，短期动能偏强。";
const S_IND = "指标";
const S_VAL = "数值";
const S_ST = "状态";
const S_NORMAL = "正常";
const S_BULL = "看多";
const S_BULLISH = "多头排列";
const S_PRED = "模型预测";
const S_ARIMA = "ARIMA 未来 5 日收盘价";
const S_LSTM = "LSTM 未来 5 日";
const S_RISK = "风险提示";
const S_DISCLAIMER = "本分析仅供学习参考，不构成投资建议。市场有风险，入市需谨慎。";
const S_OSCILLATING = "震荡上行";
const S_DEMO_DATA = "演示数据";

function hashSeed(str: string): number {
  let h = 1779033703;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return h >>> 0;
}

function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function demoStockData(symbol: string, n = 240): StockData[] {
  const rand = mulberry32(hashSeed(symbol || "AAPL"));
  const out: StockData[] = [];
  let price = 100 + (hashSeed(symbol) % 12000) / 100;
  const d = new Date();
  d.setMonth(d.getMonth() - 12);
  d.setDate(1);
  while (out.length < n) {
    const day = d.getDay();
    if (day !== 0 && day !== 6) {
      const change = (rand() - 0.48) * 0.03 + 0.0006;
      const open = price;
      const close = price * (1 + change);
      const high = Math.max(open, close) * (1 + rand() * 0.012);
      const low = Math.min(open, close) * (1 - rand() * 0.012);
      const volume = Math.round(2e7 + rand() * 8e7);
      out.push({
        date: d.toISOString().slice(0, 10),
        open: +open.toFixed(2),
        high: +high.toFixed(2),
        low: +low.toFixed(2),
        close: +close.toFixed(2),
        volume,
      });
      price = close;
    }
    d.setDate(d.getDate() + 1);
  }
  return out;
}

export function demoPredictions(symbol: string) {
  const series = demoStockData(symbol, 30);
  const last = series[series.length - 1]?.close ?? 100;
  const prand = mulberry32(hashSeed(symbol + ":pred"));
  let walk = last;
  const arima: number[] = [];
  for (let i = 0; i < 5; i++) {
    walk = walk * (1 + (prand() - 0.47) * 0.02);
    arima.push(+walk.toFixed(2));
  }
  const lrand = mulberry32(hashSeed(symbol + ":lstm"));
  let lwalk = last;
  const lstmArr: number[] = [];
  for (let i = 0; i < 5; i++) {
    lwalk = lwalk * (1 + (lrand() - 0.46) * 0.018);
    lstmArr.push(+lwalk.toFixed(2));
  }
  return { arima, lstm: lstmArr, last };
}

export function demoAnalysis(symbol: string): AnalysisResult {
  const { arima, lstm } = demoPredictions(symbol);
  const arimaLine = arima.map(v => `${v}`).join(" → ");
  const lstmLine = lstm.map(v => `${v}`).join(" → ");
  const summary = [
    `# ${symbol} ${S_DEEP_ANALYSIS}`,
    "",
    `## ${S_TREND}`,
    S_TREND_BODY,
    "",
    `| ${S_IND} | ${S_VAL} | ${S_ST} |`,
    "|---|---|---|",
    `| RSI(14) | 56.4 | ${S_NORMAL} |`,
    `| MACD | 1.24 | ${S_BULL} |`,
    `| MA5 / MA10 / MA20 | 152.34 / 150.12 / 146.88 | ${S_BULLISH} |`,
    "",
    `## ${S_PRED}`,
    `- **${S_ARIMA}**?${arimaLine}??? **${arima[arima.length - 1]}**?`,
    `- **${S_LSTM}**?${lstmLine}（末值 **${lstm[lstm.length - 1]}**）`,
    "",
    `## ${S_RISK}`,
    `> ${S_DISCLAIMER}`,
  ].join("\n");
  return {
    summary,
    trend: S_OSCILLATING,
    indicators: { ma5: 152.34, ma10: 150.12, ma20: 146.88, rsi: 56.4, macd: 1.24 },
    prediction: { arima, lstm },
    advice: S_DEMO_DATA,
    risk: S_DEMO_DATA,
  };
}

export async function backend<T>(
  cmd: string,
  args?: Record<string, unknown>,
): Promise<T> {
  if (isTauri) {
    return invoke<T>(cmd, args);
  }
  const symbol = String(args?.symbol ?? lastDemoSymbol);
  const { arima, lstm } = demoPredictions(symbol);
  switch (cmd) {
    case "fetch_stock_data":
      lastDemoSymbol = symbol;
      return demoStockData(symbol) as T;
    case "calculate_indicators":
      return { ma5: 152.34, ma10: 150.12, ma20: 146.88, rsi: 56.4, macd: 1.24 } as T;
    case "predict_arima":
      return arima as T;
    case "predict_lstm":
      return lstm as T;
    case "analyze_with_ai":
      return demoAnalysis(lastDemoSymbol) as T;
    case "fetch_models":
      return [
        { id: "deepseek-chat", name: "DeepSeek Chat", owned_by: "deepseek" },
        { id: "deepseek-coder", name: "DeepSeek Coder", owned_by: "deepseek" },
      ] as T;
    default:
      throw new Error(`[demo] unknown command: ${cmd}`);
  }
}
