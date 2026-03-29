"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownProps {
  children: string;
  className?: string;
}

export default function Markdown({ children, className = "" }: MarkdownProps) {
  return (
    <div className={`prose-rag ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => (
            <p className="text-slate-100 leading-relaxed mb-2 last:mb-0">{children}</p>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-white">{children}</strong>
          ),
          em: ({ children }) => (
            <em className="italic text-slate-200">{children}</em>
          ),
          h1: ({ children }) => (
            <h1 className="text-lg font-bold text-white mt-3 mb-1">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-base font-bold text-white mt-3 mb-1">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-sm font-semibold text-slate-100 mt-2 mb-1">{children}</h3>
          ),
          ul: ({ children }) => (
            <ul className="list-disc list-inside space-y-1 text-slate-200 my-2">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-inside space-y-1 text-slate-200 my-2">{children}</ol>
          ),
          li: ({ children }) => (
            <li className="text-sm text-slate-200 leading-relaxed">{children}</li>
          ),
          code: ({ children, className }) => {
            const isBlock = className?.startsWith("language-");
            return isBlock ? (
              <code className="block bg-slate-900 text-emerald-300 text-xs rounded-md p-3 my-2 overflow-x-auto font-mono">
                {children}
              </code>
            ) : (
              <code className="bg-slate-800 text-emerald-300 text-xs rounded px-1.5 py-0.5 font-mono">
                {children}
              </code>
            );
          },
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-slate-500 pl-3 my-2 text-slate-400 italic">
              {children}
            </blockquote>
          ),
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-400 hover:text-blue-300 underline underline-offset-2"
            >
              {children}
            </a>
          ),
          hr: () => <hr className="border-slate-700 my-3" />,
          table: ({ children }) => (
            <div className="overflow-x-auto my-2">
              <table className="text-sm w-full border-collapse">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th className="text-left text-xs font-semibold text-slate-400 uppercase tracking-wider border-b border-slate-700 pb-1 px-2">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="text-slate-200 border-b border-slate-800 py-1.5 px-2">
              {children}
            </td>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
