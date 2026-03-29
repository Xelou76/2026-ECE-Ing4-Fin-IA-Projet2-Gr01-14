export interface SourceReference {
  type: string; // "news" | "earnings" | "macro"
  ticker: string | null;
  date: string | null;
  detail: string | null;
  verified: boolean;
  badge: string; // "✅" or "⚠️"
}

export interface SimpleAnswer {
  _mode: "simple";
  answer: string;
  sources: SourceReference[];
  confidence: "high" | "medium" | "low";
}

export interface AnalystAnswer {
  _mode: "analyst";
  answer: string;
  bull_case: string;
  bear_case: string;
  risks: string[];
  catalysts: string[];
  key_metrics: Record<string, string>;
  sources: SourceReference[];
  confidence: "high" | "medium" | "low";
  signal: "bullish" | "bearish" | "neutral";
}

export type RAGAnswer = SimpleAnswer | AnalystAnswer;

export interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  question: string;
  mode: "simple" | "analyst";
  ticker?: string;
  history?: HistoryMessage[];
}

export interface StatsResponse {
  news: { collection: string; count: number };
  fundamentals: { collection: string; count: number };
  macro: { collection: string; count: number };
}

export interface HealthResponse {
  status: string;
  model: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: RAGAnswer | string;
  timestamp: Date;
}
