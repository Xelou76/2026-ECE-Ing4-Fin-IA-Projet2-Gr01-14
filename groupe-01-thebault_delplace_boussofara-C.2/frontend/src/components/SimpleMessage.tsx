"use client";

import type { SimpleAnswer } from "@/types/rag";
import Markdown from "./Markdown";
import SourceList from "./SourceList";

interface SimpleMessageProps {
  data: SimpleAnswer;
}

const CONFIDENCE_CONFIG = {
  high: { label: "High Confidence", dot: "bg-emerald-400", text: "text-emerald-400" },
  medium: { label: "Medium Confidence", dot: "bg-yellow-400", text: "text-yellow-400" },
  low: { label: "Low Confidence", dot: "bg-red-400", text: "text-red-400" },
};

export default function SimpleMessage({ data }: SimpleMessageProps) {
  const conf = CONFIDENCE_CONFIG[data.confidence] ?? CONFIDENCE_CONFIG.medium;

  return (
    <div className="space-y-3">
      {/* Answer */}
      <Markdown>{data.answer}</Markdown>

      {/* Confidence badge */}
      <div className="flex items-center gap-1.5">
        <span className={`w-2 h-2 rounded-full ${conf.dot}`} />
        <span className={`text-xs font-medium ${conf.text}`}>
          {conf.label}
        </span>
      </div>

      {/* Sources */}
      <SourceList sources={data.sources} />
    </div>
  );
}
