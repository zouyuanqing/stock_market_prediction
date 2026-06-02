import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import { StockData } from "../types";

interface StockChartProps {
  data: StockData[];
  title: string;
}

export function StockChart({ data, title }: StockChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (chartRef.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }

    return () => {
      chartInstance.current?.dispose();
    };
  }, []);

  useEffect(() => {
    if (!chartInstance.current || data.length === 0) return;

    const dates = data.map((d) => d.date);
    const closes = data.map((d) => d.close);
    const volumes = data.map((d) => d.volume);

    const option: echarts.EChartsOption = {
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "cross",
        },
      },
      legend: {
        data: ["收盘价", "成交量"],
        top: 10,
      },
      grid: [
        {
          left: "10%",
          right: "8%",
          height: "50%",
        },
        {
          left: "10%",
          right: "8%",
          top: "70%",
          height: "20%",
        },
      ],
      xAxis: [
        {
          type: "category",
          data: dates,
          gridIndex: 0,
          axisLabel: {
            show: false,
          },
        },
        {
          type: "category",
          data: dates,
          gridIndex: 1,
          axisLabel: {
            rotate: 45,
            fontSize: 10,
          },
        },
      ],
      yAxis: [
        {
          type: "value",
          name: "价格",
          gridIndex: 0,
          splitLine: {
            lineStyle: {
              type: "dashed",
            },
          },
        },
        {
          type: "value",
          name: "成交量",
          gridIndex: 1,
          splitLine: {
            show: false,
          },
        },
      ],
      series: [
        {
          name: "收盘价",
          type: "line",
          data: closes,
          xAxisIndex: 0,
          yAxisIndex: 0,
          smooth: true,
          lineStyle: {
            width: 2,
            color: "#3b82f6",
          },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: "rgba(59, 130, 246, 0.3)" },
              { offset: 1, color: "rgba(59, 130, 246, 0.05)" },
            ]),
          },
          itemStyle: {
            color: "#3b82f6",
          },
        },
        {
          name: "成交量",
          type: "bar",
          data: volumes,
          xAxisIndex: 1,
          yAxisIndex: 1,
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: "#10b981" },
              { offset: 1, color: "#059669" },
            ]),
          },
        },
      ],
    };

    chartInstance.current.setOption(option);
  }, [data]);

  useEffect(() => {
    const handleResize = () => {
      chartInstance.current?.resize();
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6">
      <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">
        {title}
      </h3>
      <div ref={chartRef} style={{ width: "100%", height: "400px" }} />
    </div>
  );
}
