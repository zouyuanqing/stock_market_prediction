"""
LLM驱动的Agent模块 - 实现自主决策的股票分析Agent
来源声明: 设计灵感来自 OpenCLAW 和 Hermes Agent 项目
- OpenCLAW: https://github.com/OpenCLAW/OpenCLAW (Apache 2.0)
- Hermes Agent: https://github.com/NousResearch/Hermes-Function-Calling (MIT)

本实现采用ReAct (Reasoning + Acting) 范式，让LLM可以:
1. 自主选择工具
2. 决定模型参数
3. 进行多步推理
4. 搜索外部信息
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from stock_market_prediction.tools import get_tool, list_tools


class StockAnalysisAgent:
    """
    LLM驱动的股票分析Agent
    
    该Agent可以:
    - 自主选择使用哪些工具
    - 决定ARIMA/LSTM的超参数
    - 搜索相关新闻和市场信息
    - 进行多步推理分析
    """
    
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.api_base = "https://api.deepseek.com"
        
        # 可用工具
        self.available_tools = list_tools()
        self.tool_names = [t["name"] for t in self.available_tools]
        
        # 对话历史
        self.conversation_history = []
        
        # 系统提示词
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        
        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in self.available_tools
        ])
        
        return f"""你是一个专业的股票分析AI助手。你可以使用以下工具来分析股票:

{tools_description}

## 重要规则:

1. **工具调用格式**: 当你需要使用工具时，必须严格按照以下JSON格式输出:
```json
{{
    "action": "tool_name",
    "action_input": {{
        "param1": "value1",
        "param2": "value2"
    }}
}}
```

2. **ARIMA参数选择指南**:
   - p (AR阶数): 通常1-3，如果数据有明显趋势用较高值
   - d (差分阶数): 通常1，使序列平稳
   - q (MA阶数): 通常1-2
   - 建议先尝试(1,1,1)，根据AIC/BIC调整

3. **LSTM参数选择指南**:
   - hidden_size: 32-128，复杂模式用较大值
   - num_layers: 1-3层，深层网络需要更多数据
   - lookback_days: 30-60天，取决于预测周期
   - epochs: 30-100，避免过拟合

4. **分析流程建议**:
   a. 先获取股票历史数据
   b. 搜索相关新闻和市场动态
   c. 计算技术指标
   d. 运行ARIMA和LSTM预测
   e. 综合所有信息给出投资建议

5. **最终回答格式**: 完成分析后，用自然语言总结你的发现和建议。

记住: 你必须基于数据和事实进行分析，不要编造信息。
"""

    def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """调用LLM API"""
        import requests
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def _parse_tool_call(self, llm_response: str) -> Optional[Tuple[str, Dict]]:
        """解析LLM的工具调用请求"""
        
        # 尝试查找JSON格式的工具调用
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, llm_response, re.DOTALL)
        
        if not matches:
            # 尝试直接查找JSON对象
            json_pattern2 = r'\{[^{}]*"action"[^{}]*\}'
            matches = re.findall(json_pattern2, llm_response, re.DOTALL)
        
        if matches:
            try:
                tool_call = json.loads(matches[0])
                if "action" in tool_call and "action_input" in tool_call:
                    return tool_call["action"], tool_call["action_input"]
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _execute_tool(self, tool_name: str, params: Dict) -> Dict[str, Any]:
        """执行工具"""
        try:
            tool = get_tool(tool_name)
            result = tool.execute(**params)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def analyze(self, 
                ticker: str,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                max_iterations: int = 10) -> Dict[str, Any]:
        """
        执行完整的股票分析
        
        Args:
            ticker: 股票代码
            start_date: 开始日期，默认1年前
            end_date: 结束日期，默认今天
            max_iterations: 最大工具调用次数
        
        Returns:
            分析结果
        """
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # 初始化对话
        user_query = f"请分析股票 {ticker}，时间范围从 {start_date} 到 {end_date}。我需要全面的分析包括价格走势、技术指标和预测。"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        analysis_results = {
            "ticker": ticker,
            "date_range": {"start": start_date, "end": end_date},
            "tool_calls": [],
            "final_analysis": None,
            "iterations": 0
        }
        
        print(f"\n🚀 开始分析 {ticker}...")
        print("=" * 60)
        
        for iteration in range(max_iterations):
            analysis_results["iterations"] = iteration + 1
            
            # 调用LLM
            print(f"\n[Iteration {iteration + 1}] 思考中...")
            llm_response = self._call_llm(messages)
            
            print(f"LLM响应:\n{llm_response[:500]}...")
            
            # 解析工具调用
            tool_call = self._parse_tool_call(llm_response)
            
            if tool_call is None:
                # LLM给出了最终答案
                print("\n✅ 分析完成!")
                analysis_results["final_analysis"] = llm_response
                break
            
            tool_name, params = tool_call
            print(f"\n🔧 执行工具: {tool_name}")
            print(f"参数: {params}")
            
            # 添加LLM响应到历史
            messages.append({"role": "assistant", "content": llm_response})
            
            # 执行工具
            tool_result = self._execute_tool(tool_name, params)
            
            # 记录工具调用
            analysis_results["tool_calls"].append({
                "iteration": iteration + 1,
                "tool": tool_name,
                "params": params,
                "result": tool_result
            })
            
            # 添加工具结果到对话
            if tool_result["success"]:
                result_summary = json.dumps(tool_result["result"], indent=2)[:1000]
                tool_message = f"工具 {tool_name} 执行成功:\n{result_summary}"
            else:
                tool_message = f"工具 {tool_name} 执行失败: {tool_result.get('error', 'Unknown error')}"
            
            messages.append({"role": "user", "content": tool_message})
            print(f"工具结果: {tool_message[:200]}...")
        
        # 如果没有得到最终答案，强制总结
        if analysis_results["final_analysis"] is None:
            print("\n⚠️ 达到最大迭代次数，生成总结...")
            summary_prompt = "基于以上所有工具调用的结果，请给出对股票的全面分析和投资建议。"
            messages.append({"role": "user", "content": summary_prompt})
            analysis_results["final_analysis"] = self._call_llm(messages)
        
        print("\n" + "=" * 60)
        print("📊 分析完成!")
        
        return analysis_results
    
    def run_interactive(self, ticker: str):
        """交互式分析模式"""
        print(f"\n🤖 启动 {ticker} 的交互式分析...")
        print("输入 'quit' 退出，输入 'analyze' 开始自动分析")
        
        while True:
            user_input = input("\n👤 你: ").strip()
            
            if user_input.lower() == 'quit':
                print("再见!")
                break
            
            if user_input.lower() == 'analyze':
                results = self.analyze(ticker)
                print("\n📈 分析结果:")
                print(results["final_analysis"])
                continue
            
            # 直接调用LLM
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            response = self._call_llm(messages)
            print(f"\n🤖 Agent: {response}")


def main():
    """主函数示例"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("❌ 错误: 未找到 DEEPSEEK_API_KEY")
        print("请在 .env 文件中设置 DEEPSEEK_API_KEY=your_key")
        return
    
    # 创建Agent
    agent = StockAnalysisAgent(api_key=api_key)
    
    # 分析示例股票
    ticker = "AAPL"
    print(f"\n开始分析 {ticker}...")
    
    results = agent.analyze(ticker)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("📊 最终分析报告")
    print("=" * 80)
    print(results["final_analysis"])
    
    # 保存结果
    output_file = f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        # 移除可能的大数据以便序列化
        serializable_results = results.copy()
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\n💾 详细结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
