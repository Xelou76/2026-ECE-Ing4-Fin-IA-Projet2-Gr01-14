"use client";

import type { AnalystAnswer } from "@/types/rag";
import Markdown from "./Markdown";
import SourceList from "./SourceList";

interface AnalystCardProps {
  data: AnalystAnswer;
}

const SIGNAL_CONFIG = {
  bullish: {
    banner: "bg-emerald-900/50 border-emerald-600/60",
    icon: "🟢",
    label: "BULLISH",
    text: "text-emerald-300",
  },
  bearish: {
    banner: "bg-red-900/50 border-red-600/60",
    icon: "🔴",
    label: "BEARISH",
    text: "text-red-300",
  },
  neutral: {
    banner: "bg-slate-700/50 border-slate-600/60",
    icon: "⚪",
    label: "NEUTRAL",
    text: "text-slate-300",
  },
};

const CONFIDENCE_CONFIG = {
  high: { dot: "bg-emerald-400", text: "text-emerald-400", label: "High Confidence" },
  medium: { dot: "bg-yellow-400", text: "text-yellow-400", label: "Medium Confidence" },
  low: { dot: "bg-red-400", text: "text-red-400", label: "Low Confidence" },
};

export default function AnalystCard({ data }: AnalystCardProps) {
  const signal = SIGNAL_CONFIG[data.signal] ?? SIGNAL_CONFIG.neutral;
  const conf = CONFIDENCE_CONFIG[data.confidence] ?? CONFIDENCE_CONFIG.medium;

  return (
    <div className="space-y-4">
      {/* Signal banner */}
      <div
        className={`flex items-center gap-2 px-4 py-2 rounded-lg border ${signal.banner}`}
      >
        <span className="text-lg">{signal.icon}</span>
        <span className={`font-bold text-base tracking-widest ${signal.text}`}>
          {signal.label}
        </span>
      </div>

      {/* Executive summary */}
      <Markdown>{data.answer}</Markdown>

      {/* Bull / Bear columns */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {/* Bull case */}
        <div className="bg-emerald-900/20 border border-emerald-800/40 rounded-lg p-3">
          <p className="text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1.5">
            Bull Case
          </p>
          <Markdown className="text-sm">{data.bull_case}</Markdown>
        </div>

        {/* Bear case */}
        <div className="bg-red-900/20 border border-red-800/40 rounded-lg p-3">
          <p className="text-xs font-semibold text-red-400 uppercase tracking-wider mb-1.5">
            Bear Case
          </p>
          <Markdown className="text-sm">{data.bear_case}</Markdown>
        </div>
      </div>

      {/* Key Metrics */}
      {data.key_metrics && Object.keys(data.key_metrics).length > 0 && (
        <div>
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
            Key Metrics
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {Object.entries(data.key_metrics).map(([key, value]) => (
              <div
                key={key}
                className="bg-slate-800/60 border border-slate-700/50 rounded-md px-3 py-2"
              >
                <p className="text-[10px] text-slate-500 uppercase tracking-wider">
                  {key.replace(/_/g, " ")}
                </p>
                <p className="font-mono text-sm font-semibold text-slate-100 mt-0.5">
                  {value}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Catalysts & Risks */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {/* Catalysts */}
        {data.catalysts && data.catalysts.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Catalysts
            </p>
            <ul className="space-y-1">
              {data.catalysts.map((c, i) => (
                <li key={i} className="flex gap-2 text-sm text-slate-300">
                  <span className="text-emerald-400 shrink-0">+</span>
                  <span>{c}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Risks */}
        {data.risks && data.risks.length > 0 && (
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Risks
            </p>
            <ul className="space-y-1">
              {data.risks.map((r, i) => (
                <li key={i} className="flex gap-2 text-sm text-slate-300">
                  <span className="text-red-400 shrink-0">−</span>
                  <span>{r}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Confidence */}
      <div className="flex items-center gap-1.5">
        <span className={`w-2 h-2 rounded-full ${conf.dot}`} />
        <span className={`text-xs font-medium ${conf.text}`}>{conf.label}</span>
      </div>

      {/* Sources */}
      <SourceList sources={data.sources} />
    </div>
  );
}
