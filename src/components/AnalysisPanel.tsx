import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AnalysisResult } from "../types";

interface AnalysisPanelProps {
  analysis: AnalysisResult;
}

export function AnalysisPanel({ analysis }: AnalysisPanelProps) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6">
      <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">
        🤖 AI 深度分析
      </h3>

      <div className="prose prose-blue dark:prose-invert max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            // 自定义标题样式
            h1: ({ children }) => (
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-4 pb-2 border-b">
                {children}
              </h1>
            ),
            h2: ({ children }) => (
              <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-3 mt-6">
                {children}
              </h2>
            ),
            h3: ({ children }) => (
              <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2 mt-4">
                {children}
              </h3>
            ),
            // 自定义段落样式
            p: ({ children }) => (
              <p className="text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                {children}
              </p>
            ),
            // 自定义列表样式
            ul: ({ children }) => (
              <ul className="list-disc list-inside space-y-2 mb-4 text-gray-600 dark:text-gray-400">
                {children}
              </ul>
            ),
            ol: ({ children }) => (
              <ol className="list-decimal list-inside space-y-2 mb-4 text-gray-600 dark:text-gray-400">
                {children}
              </ol>
            ),
            li: ({ children }) => (
              <li className="ml-4">{children}</li>
            ),
            // 自定义表格样式
            table: ({ children }) => (
              <div className="overflow-x-auto mb-4">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  {children}
                </table>
              </div>
            ),
            thead: ({ children }) => (
              <thead className="bg-gray-50 dark:bg-gray-700">{children}</thead>
            ),
            th: ({ children }) => (
              <th className="px-4 py-2 text-left text-sm font-medium text-gray-700 dark:text-gray-300">
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td className="px-4 py-2 text-sm text-gray-600 dark:text-gray-400">
                {children}
              </td>
            ),
            // 自定义代码块样式
            code: ({ children, className }) => {
              const isInline = !className;
              if (isInline) {
                return (
                  <code className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm font-mono text-blue-600 dark:text-blue-400">
                    {children}
                  </code>
                );
              }
              return (
                <code className={className}>{children}</code>
              );
            },
            pre: ({ children }) => (
              <pre className="bg-gray-900 text-gray-100 rounded-lg p-4 mb-4 overflow-x-auto">
                {children}
              </pre>
            ),
            // 自定义引用块样式
            blockquote: ({ children }) => (
              <blockquote className="border-l-4 border-blue-500 pl-4 py-2 mb-4 bg-blue-50 dark:bg-blue-900/20 rounded-r-lg">
                {children}
              </blockquote>
            ),
            // 自定义强调样式
            strong: ({ children }) => (
              <strong className="font-semibold text-gray-800 dark:text-gray-200">
                {children}
              </strong>
            ),
            em: ({ children }) => (
              <em className="italic text-gray-700 dark:text-gray-300">
                {children}
              </em>
            ),
            // 自定义分隔线
            hr: () => (
              <hr className="my-6 border-gray-200 dark:border-gray-700" />
            ),
          }}
        >
          {analysis.summary}
        </ReactMarkdown>
      </div>

      {/* 免责声明 */}
      <div className="mt-6 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
        <p className="text-yellow-800 dark:text-yellow-200 text-sm">
          ⚠️ <strong>免责声明：</strong>本分析仅供参考，不构成投资建议。投资有风险，入市需谨慎。
        </p>
      </div>
    </div>
  );
}
