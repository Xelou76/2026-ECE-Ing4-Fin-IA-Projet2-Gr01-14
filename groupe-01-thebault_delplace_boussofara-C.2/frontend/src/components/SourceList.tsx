"use client";

import type { SourceReference } from "@/types/rag";

interface SourceListProps {
  sources: SourceReference[];
}

const TYPE_COLORS: Record<string, string> = {
  news: "bg-blue-900/50 text-blue-300 border-blue-700/50",
  earnings: "bg-purple-900/50 text-purple-300 border-purple-700/50",
  macro: "bg-amber-900/50 text-amber-300 border-amber-700/50",
};

export default function SourceList({ sources }: SourceListProps) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-3 pt-3 border-t border-slate-700/60">
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
        Sources ({sources.length})
      </p>
      <ul className="space-y-1.5">
        {sources.map((src, idx) => (
          <li
            key={idx}
            className="flex items-start gap-2 text-xs text-slate-400"
          >
            <span className="shrink-0 mt-0.5">{src.badge}</span>
            <span
              className={`shrink-0 px-1.5 py-0.5 rounded border text-[10px] font-mono font-semibold uppercase ${
                TYPE_COLORS[src.type] ?? "bg-slate-700 text-slate-300 border-slate-600"
              }`}
            >
              {src.type}
            </span>
            {src.ticker && (
              <span className="shrink-0 font-mono font-bold text-slate-200">
                {src.ticker}
              </span>
            )}
            {src.date && (
              <span className="shrink-0 text-slate-500">{src.date}</span>
            )}
            {src.detail && (
              <span className="text-slate-400 truncate" title={src.detail}>
                {src.detail.length > 80
                  ? src.detail.slice(0, 80) + "…"
                  : src.detail}
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
