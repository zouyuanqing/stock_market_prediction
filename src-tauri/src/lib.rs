use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::process::Command as AsyncCommand;

// 股票数据结构
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StockData {
    pub date: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
}

// 技术指标结构
#[derive(Debug, Serialize, Deserialize)]
pub struct TechnicalIndicators {
    pub ma5: f64,
    pub ma10: f64,
    pub ma20: f64,
    pub rsi: f64,
    pub macd: f64,
}

// 预测结果结构
#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionResult {
    pub arima: Vec<f64>,
    pub lstm: f64,
}

// AI 分析结果结构
#[derive(Debug, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub summary: String,
    pub trend: String,
    pub indicators: TechnicalIndicators,
    pub prediction: PredictionResult,
    pub advice: String,
    pub risk: String,
}

// 模型信息结构
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub owned_by: String,
}

// 辅助函数：将 anyhow::Error 转换为 String
fn to_string_error<T>(result: anyhow::Result<T>) -> Result<T, String> {
    result.map_err(|e| format!("{:#}", e))
}

// 获取 Python 脚本目录
fn get_python_script_dir() -> PathBuf {
    // 开发模式：相对于 src-tauri/python/
    let mut path = std::env::current_dir().unwrap_or_default();
    path.push("src-tauri");
    path.push("python");
    path
}

// 检测 Python 命令
async fn find_python_command() -> anyhow::Result<String> {
    // 尝试 python
    let output = AsyncCommand::new("python")
        .arg("--version")
        .output()
        .await;

    if output.is_ok() {
        return Ok("python".to_string());
    }

    // 尝试 python3
    let output = AsyncCommand::new("python3")
        .arg("--version")
        .output()
        .await;

    if output.is_ok() {
        return Ok("python3".to_string());
    }

    Err(anyhow::anyhow!(
        "Python not found. Please install Python 3.9+ and add it to PATH."
    ))
}

// 调用 Python 脚本的通用函数
async fn call_python_script(script_name: &str, input: serde_json::Value) -> anyhow::Result<serde_json::Value> {
    let python_cmd = find_python_command().await?;
    let script_path = get_python_script_dir().join(script_name);

    if !script_path.exists() {
        return Err(anyhow::anyhow!("Script not found: {:?}", script_path));
    }

    let mut child = AsyncCommand::new(&python_cmd)
        .arg(&script_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context("Failed to spawn Python process")?;

    // 写入 stdin
    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        let input_str = input.to_string();
        stdin.write_all(input_str.as_bytes()).await?;
        stdin.shutdown().await?;
    }

    // 等待完成（带超时）
    let output = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        child.wait_with_output(),
    )
    .await
    .context("Python script timed out (30s)")?
    .context("Failed to wait for Python process")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("Python script failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .context("Failed to parse Python output as JSON")?;

    // 检查是否有错误
    if let Some(error) = parsed.get("error") {
        return Err(anyhow::anyhow!("Python error: {}", error));
    }

    Ok(parsed)
}

// 获取股票数据
#[tauri::command]
async fn fetch_stock_data(
    symbol: String,
    start_date: String,
    end_date: String,
    source: String,
) -> Result<Vec<StockData>, String> {
    to_string_error(fetch_stock_data_inner(symbol, start_date, end_date, source).await)
}

async fn fetch_stock_data_inner(
    symbol: String,
    start_date: String,
    end_date: String,
    source: String,
) -> anyhow::Result<Vec<StockData>> {
    match source.as_str() {
        "yahoo" => fetch_from_yahoo(&symbol, &start_date, &end_date).await,
        "stooq" => fetch_from_stooq(&symbol, &start_date, &end_date).await,
        _ => Err(anyhow::anyhow!("Unsupported data source: {}", source)),
    }
}

// 从 Yahoo Finance 获取数据
async fn fetch_from_yahoo(
    symbol: &str,
    start_date: &str,
    end_date: &str,
) -> anyhow::Result<Vec<StockData>> {
    let client = reqwest::Client::new();

    let start_timestamp = chrono::NaiveDate::parse_from_str(start_date, "%Y-%m-%d")
        .context("Invalid start date format")?
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp();

    let end_timestamp = chrono::NaiveDate::parse_from_str(end_date, "%Y-%m-%d")
        .context("Invalid end date format")?
        .and_hms_opt(23, 59, 59)
        .unwrap()
        .and_utc()
        .timestamp();

    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{}?period1={}&period2={}&interval=1d",
        symbol, start_timestamp, end_timestamp
    );

    let response = client
        .get(&url)
        .header("User-Agent", "Mozilla/5.0")
        .send()
        .await
        .context("Failed to fetch data from Yahoo Finance")?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!("HTTP error: {}", response.status()));
    }

    let data: serde_json::Value = response
        .json()
        .await
        .context("Failed to parse Yahoo Finance response")?;

    let mut stocks = Vec::new();

    if let Some(result) = data["chart"]["result"].as_array() {
        if let Some(first) = result.first() {
            if let (Some(timestamps), Some(quotes)) = (
                first["timestamp"].as_array(),
                first["indicators"]["quote"].as_array(),
            ) {
                if let Some(quote) = quotes.first() {
                    let default_vec = vec![];
                    let opens = quote["open"].as_array().unwrap_or(&default_vec);
                    let highs = quote["high"].as_array().unwrap_or(&default_vec);
                    let lows = quote["low"].as_array().unwrap_or(&default_vec);
                    let closes = quote["close"].as_array().unwrap_or(&default_vec);
                    let volumes = quote["volume"].as_array().unwrap_or(&default_vec);

                    for (i, timestamp) in timestamps.iter().enumerate() {
                        if let Some(ts) = timestamp.as_i64() {
                            let date = chrono::DateTime::from_timestamp(ts, 0)
                                .map(|dt| dt.format("%Y-%m-%d").to_string())
                                .unwrap_or_default();

                            let open = opens.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let high = highs.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let low = lows.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let close = closes.get(i).and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let volume = volumes.get(i).and_then(|v| v.as_u64()).unwrap_or(0);

                            stocks.push(StockData {
                                date,
                                open,
                                high,
                                low,
                                close,
                                volume,
                            });
                        }
                    }
                }
            }
        }
    }

    Ok(stocks)
}

// 从 Stooq 获取数据
async fn fetch_from_stooq(
    symbol: &str,
    start_date: &str,
    end_date: &str,
) -> anyhow::Result<Vec<StockData>> {
    let formatted_symbol = symbol.to_lowercase();
    let url = format!(
        "https://stooq.com/q/d/l/?s={}&d1={}&d2={}&i=d",
        formatted_symbol,
        start_date.replace("-", ""),
        end_date.replace("-", "")
    );

    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header("User-Agent", "Mozilla/5.0")
        .send()
        .await
        .context("Failed to fetch data from Stooq")?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!("HTTP error: {}", response.status()));
    }

    let text = response
        .text()
        .await
        .context("Failed to read Stooq response")?;

    let mut stocks = Vec::new();
    let lines: Vec<&str> = text.lines().collect();

    for line in lines.iter().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 6 {
            let date = parts[0].to_string();
            let open = parts[1].parse::<f64>().unwrap_or(0.0);
            let high = parts[2].parse::<f64>().unwrap_or(0.0);
            let low = parts[3].parse::<f64>().unwrap_or(0.0);
            let close = parts[4].parse::<f64>().unwrap_or(0.0);
            let volume = parts[5].parse::<u64>().unwrap_or(0);

            stocks.push(StockData {
                date,
                open,
                high,
                low,
                close,
                volume,
            });
        }
    }

    Ok(stocks)
}

// 计算技术指标
#[tauri::command]
async fn calculate_indicators(data: Vec<StockData>) -> Result<TechnicalIndicators, String> {
    to_string_error(calculate_indicators_inner(data))
}

fn calculate_indicators_inner(data: Vec<StockData>) -> anyhow::Result<TechnicalIndicators> {
    if data.len() < 20 {
        return Err(anyhow::anyhow!(
            "Not enough data to calculate indicators (need at least 20 days)"
        ));
    }

    let closes: Vec<f64> = data.iter().map(|d| d.close).collect();
    let len = closes.len();

    let ma5 = closes[len - 5..].iter().sum::<f64>() / 5.0;
    let ma10 = closes[len - 10..].iter().sum::<f64>() / 10.0;
    let ma20 = closes[len - 20..].iter().sum::<f64>() / 20.0;

    let mut gains = Vec::new();
    let mut losses = Vec::new();
    for i in 1..len {
        let change = closes[i] - closes[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    let avg_gain = gains.iter().sum::<f64>() / gains.len() as f64;
    let avg_loss = losses.iter().sum::<f64>() / losses.len() as f64;
    let rs = if avg_loss == 0.0 {
        100.0
    } else {
        avg_gain / avg_loss
    };
    let rsi = 100.0 - (100.0 / (1.0 + rs));

    let ema12 = calculate_ema(&closes, 12);
    let ema26 = calculate_ema(&closes, 26);
    let macd = ema12 - ema26;

    Ok(TechnicalIndicators {
        ma5,
        ma10,
        ma20,
        rsi,
        macd,
    })
}

fn calculate_ema(data: &[f64], period: usize) -> f64 {
    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema = data[0];

    for i in 1..data.len() {
        ema = (data[i] - ema) * multiplier + ema;
    }

    ema
}

// ARIMA 预测（调用 Python sidecar）
#[tauri::command]
async fn predict_arima(data: Vec<StockData>, days: usize) -> Result<Vec<f64>, String> {
    to_string_error(predict_arima_inner(data, days).await)
}

async fn predict_arima_inner(data: Vec<StockData>, days: usize) -> anyhow::Result<Vec<f64>> {
    if data.len() < 30 {
        return Err(anyhow::anyhow!(
            "Need at least 30 data points for ARIMA prediction"
        ));
    }

    let dates: Vec<String> = data.iter().map(|d| d.date.clone()).collect();
    let prices: Vec<f64> = data.iter().map(|d| d.close).collect();

    let input = serde_json::json!({
        "dates": dates,
        "prices": prices,
        "days": days
    });

    let result = call_python_script("predict_arima.py", input).await?;

    let predictions: Vec<f64> = result["predictions"]
        .as_array()
        .context("Missing predictions in Python output")?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0))
        .collect();

    Ok(predictions)
}

// LSTM 预测（调用 Python sidecar）
#[tauri::command]
async fn predict_lstm(data: Vec<StockData>) -> Result<f64, String> {
    to_string_error(predict_lstm_inner(data).await)
}

async fn predict_lstm_inner(data: Vec<StockData>) -> anyhow::Result<f64> {
    if data.len() < 60 {
        return Err(anyhow::anyhow!(
            "Need at least 60 data points for LSTM prediction"
        ));
    }

    let dates: Vec<String> = data.iter().map(|d| d.date.clone()).collect();
    let prices: Vec<f64> = data.iter().map(|d| d.close).collect();

    let input = serde_json::json!({
        "dates": dates,
        "prices": prices,
        "window_size": 20,
        "epochs": 100,
        "hidden_size": 64,
        "num_layers": 2
    });

    let result = call_python_script("predict_lstm.py", input).await?;

    let prediction = result["prediction"]
        .as_f64()
        .context("Missing prediction in Python output")?;

    Ok(prediction)
}

// 获取可用模型列表
#[tauri::command]
async fn fetch_models(api_key: String, api_base: String) -> Result<Vec<ModelInfo>, String> {
    to_string_error(fetch_models_inner(api_key, api_base).await)
}

async fn fetch_models_inner(api_key: String, api_base: String) -> anyhow::Result<Vec<ModelInfo>> {
    if api_key.is_empty() {
        return Ok(vec![
            ModelInfo {
                id: "deepseek-chat".to_string(),
                name: "DeepSeek Chat".to_string(),
                owned_by: "deepseek".to_string(),
            },
            ModelInfo {
                id: "deepseek-coder".to_string(),
                name: "DeepSeek Coder".to_string(),
                owned_by: "deepseek".to_string(),
            },
        ]);
    }

    let client = reqwest::Client::new();
    let url = format!("{}/models", api_base);

    let response = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .send()
        .await
        .context("Failed to fetch models from API")?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!("API error: {}", response.status()));
    }

    let data: serde_json::Value = response
        .json()
        .await
        .context("Failed to parse models response")?;

    let mut models = Vec::new();

    if let Some(data_array) = data["data"].as_array() {
        for item in data_array {
            if let (Some(id), Some(owned_by)) = (item["id"].as_str(), item["owned_by"].as_str())
            {
                models.push(ModelInfo {
                    id: id.to_string(),
                    name: id.to_string(),
                    owned_by: owned_by.to_string(),
                });
            }
        }
    }

    Ok(models)
}

// 调用 AI 分析
#[tauri::command]
async fn analyze_with_ai(
    data: Vec<StockData>,
    api_key: String,
    api_base: String,
    model: String,
) -> Result<AnalysisResult, String> {
    to_string_error(analyze_with_ai_inner(data, api_key, api_base, model).await)
}

async fn analyze_with_ai_inner(
    data: Vec<StockData>,
    api_key: String,
    api_base: String,
    model: String,
) -> anyhow::Result<AnalysisResult> {
    // 计算技术指标
    let indicators = calculate_indicators_inner(data.clone())?;

    // 尝试获取预测（允许失败）
    let arima_predictions = match predict_arima_inner(data.clone(), 5).await {
        Ok(pred) => pred,
        Err(e) => {
            // Python 不可用时，返回空预测
            eprintln!("ARIMA prediction failed: {}", e);
            vec![0.0; 5]
        }
    };

    let lstm_prediction = match predict_lstm_inner(data.clone()).await {
        Ok(pred) => pred,
        Err(e) => {
            eprintln!("LSTM prediction failed: {}", e);
            0.0
        }
    };

    let prediction = PredictionResult {
        arima: arima_predictions.clone(),
        lstm: lstm_prediction,
    };

    if api_key.is_empty() {
        return Ok(AnalysisResult {
            summary: "⚠️ 未配置 API Key，仅显示本地技术分析结果。\n\n配置 DeepSeek 或 OpenAI API 后可获得 AI 深度分析。".to_string(),
            trend: "基于技术指标分析".to_string(),
            indicators,
            prediction,
            advice: "请配置 AI API 以获取详细操作建议".to_string(),
            risk: "投资有风险，入市需谨慎".to_string(),
        });
    }

    let client = reqwest::Client::new();

    let last_5_days: Vec<&StockData> = data.iter().rev().take(5).collect();
    let data_summary =
        serde_json::to_string(&last_5_days).unwrap_or_else(|_| "Failed to serialize data".to_string());

    let prompt = format!(
        r#"请作为专业股票分析师，分析以下股票数据并提供详细报告。

## 最近5天数据
{}

## 技术指标
- MA5: {:.2}
- MA10: {:.2}
- MA20: {:.2}
- RSI: {:.2}
- MACD: {:.4}

## ARIMA 预测（未来5天）
{:?}

## LSTM 预测
{:.2}

## 要求
请用 Markdown 格式输出，包含以下章节：

### 📊 数据概览
简述股票近期表现

### 📈 趋势分析
分析价格走势和成交量变化

### 🔧 技术指标解读
解读 MA、RSI、MACD 等指标含义

### 🔮 未来预测
结合 ARIMA 和 LSTM 预测结果给出综合预测

### 💼 操作建议
给出具体的操作建议

### ⚠️ 风险提示
列出主要风险因素"#,
        data_summary,
        indicators.ma5,
        indicators.ma10,
        indicators.ma20,
        indicators.rsi,
        indicators.macd,
        arima_predictions,
        lstm_prediction
    );

    let request_body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个专业的股票市场分析师，请用中文回复，使用 Markdown 格式。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    });

    let response = client
        .post(&format!("{}/chat/completions", api_base))
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await
        .context("Failed to call AI API")?;

    if !response.status().is_success() {
        let error_text = response.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!("AI API error: {}", error_text));
    }

    let response_json: serde_json::Value = response
        .json()
        .await
        .context("Failed to parse AI response")?;

    let analysis_text = response_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("Failed to generate analysis")
        .to_string();

    Ok(AnalysisResult {
        summary: analysis_text,
        trend: "基于技术分析".to_string(),
        indicators,
        prediction,
        advice: "详见分析报告".to_string(),
        risk: "投资有风险，入市需谨慎".to_string(),
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            fetch_stock_data,
            calculate_indicators,
            predict_arima,
            predict_lstm,
            fetch_models,
            analyze_with_ai
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
