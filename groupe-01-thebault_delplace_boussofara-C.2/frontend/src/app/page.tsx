"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import type { Message, StatsResponse, SimpleAnswer, AnalystAnswer } from "@/types/rag";
import { sendChat, getStats, getHealth } from "@/lib/api";
import ChatMessage from "@/components/ChatMessage";

// Suggested starter questions
const SUGGESTIONS = [
  "What is the latest news on NVDA?",
  "Analyze AAPL fundamentals for me",
  "What is the current macro outlook with VIX and rates?",
  "Compare MSFT and GOOGL revenue growth",
];

function generateId(): string {
  return Math.random().toString(36).slice(2, 11);
}

// Spinner component
function Spinner() {
  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-[80%]">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center text-xs font-bold text-white shrink-0">
            BV
          </div>
          <span className="text-xs font-semibold text-slate-400">
            BrightVest AI
          </span>
        </div>
        <div className="bg-slate-800 border border-slate-700/60 rounded-2xl rounded-tl-sm px-4 py-4 shadow-lg">
          <div className="flex items-center gap-2">
            <div className="flex gap-1">
              <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0ms]" />
              <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:150ms]" />
              <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:300ms]" />
            </div>
            <span className="text-xs text-slate-500">Analyzing…</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Stats footer component
function StatsFooter({ stats }: { stats: StatsResponse | null }) {
  if (!stats) return null;
  return (
    <div className="flex items-center justify-center gap-4 py-2 text-[11px] text-slate-600 border-t border-slate-800">
      <span>
        <span className="text-slate-500 font-semibold">News</span>{" "}
        {stats.news.count.toLocaleString()}
      </span>
      <span className="text-slate-700">·</span>
      <span>
        <span className="text-slate-500 font-semibold">Fundamentals</span>{" "}
        {stats.fundamentals.count.toLocaleString()}
      </span>
      <span className="text-slate-700">·</span>
      <span>
        <span className="text-slate-500 font-semibold">Macro</span>{" "}
        {stats.macro.count.toLocaleString()}
      </span>
    </div>
  );
}

export default function HomePage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<"simple" | "analyst">("simple");
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [backendStatus, setBackendStatus] = useState<"online" | "offline" | "checking">("checking");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading, scrollToBottom]);

  // Load stats & health on mount
  useEffect(() => {
    getHealth()
      .then(() => setBackendStatus("online"))
      .catch(() => setBackendStatus("offline"));

    getStats()
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  // Auto-resize textarea
  useEffect(() => {
    const ta = inputRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
  }, [input]);

  const sendMessage = useCallback(
    async (text: string) => {
      const question = text.trim();
      if (!question || loading) return;

      // Add user message
      const userMsg: Message = {
        id: generateId(),
        role: "user",
        content: question,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setInput("");
      setLoading(true);

      try {
        // Build conversation history from previous messages (last 6 = 3 turns).
        // Truncate assistant answers to 500 chars to avoid token bloat.
        const history = messages.slice(-6).map((msg) => ({
          role: msg.role,
          content:
            typeof msg.content === "string"
              ? (msg.content as string).slice(0, 500)
              : ((msg.content as SimpleAnswer | AnalystAnswer).answer ?? "").slice(0, 500),
        }));

        const response = await sendChat({
          question,
          mode,
          history: history.length > 0 ? history : undefined,
        });
        const assistantMsg: Message = {
          id: generateId(),
          role: "assistant",
          content: response,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (err) {
        const errorMsg: Message = {
          id: generateId(),
          role: "assistant",
          content: `Error: ${err instanceof Error ? err.message : "Failed to reach the API. Make sure the backend is running on localhost:8000."}`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMsg]);
      } finally {
        setLoading(false);
        setTimeout(() => inputRef.current?.focus(), 50);
      }
    },
    [loading, mode]
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-screen bg-slate-900">
      {/* ── Header ── */}
      <header className="shrink-0 border-b border-slate-800 bg-slate-900/95 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-sm font-bold text-white">
              BV
            </div>
            <div>
              <h1 className="text-sm font-bold text-slate-100 leading-none">
                BrightVest RAG
              </h1>
              <p className="text-[10px] text-slate-500 mt-0.5">
                Financial AI Assistant
              </p>
            </div>
          </div>

          {/* Right: backend status + mode toggle */}
          <div className="flex items-center gap-3">
            {/* Backend status */}
            <div className="flex items-center gap-1.5">
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  backendStatus === "online"
                    ? "bg-emerald-400"
                    : backendStatus === "offline"
                    ? "bg-red-400"
                    : "bg-yellow-400 animate-pulse"
                }`}
              />
              <span className="text-[10px] text-slate-500 hidden sm:inline">
                {backendStatus === "online"
                  ? "API Online"
                  : backendStatus === "offline"
                  ? "API Offline"
                  : "Connecting…"}
              </span>
            </div>

            {/* Mode toggle */}
            <div className="flex items-center bg-slate-800 border border-slate-700 rounded-lg p-0.5">
              <button
                onClick={() => setMode("simple")}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
                  mode === "simple"
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Simple
              </button>
              <button
                onClick={() => setMode("analyst")}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
                  mode === "analyst"
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Analyst
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* ── Messages area ── */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 py-6">
          {/* Empty state */}
          {isEmpty && (
            <div className="flex flex-col items-center justify-center min-h-[50vh] text-center">
              <div className="w-16 h-16 rounded-2xl bg-blue-600/20 border border-blue-600/30 flex items-center justify-center text-3xl mb-4">
                📊
              </div>
              <h2 className="text-xl font-semibold text-slate-200 mb-2">
                Ask a financial question
              </h2>
              <p className="text-sm text-slate-500 max-w-sm mb-8">
                Powered by RAG over{" "}
                {stats
                  ? `${(stats.news.count + stats.fundamentals.count + stats.macro.count).toLocaleString()} indexed documents`
                  : "news, fundamentals, and macro data"}
                .
              </p>

              {/* Suggestion chips */}
              <div className="flex flex-wrap gap-2 justify-center max-w-lg">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => sendMessage(s)}
                    className="px-3 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 hover:border-slate-600 rounded-xl text-xs text-slate-300 hover:text-slate-100 transition-all text-left"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Message list */}
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}

          {/* Loading spinner */}
          {loading && <Spinner />}

          {/* Scroll anchor */}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* ── Input area ── */}
      <div className="shrink-0 border-t border-slate-800 bg-slate-900/95 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto px-4 py-3">
          <div className="flex items-end gap-2 bg-slate-800 border border-slate-700 rounded-2xl px-4 py-2 focus-within:border-blue-500/60 transition-colors">
            {/* Mode badge inside input */}
            <span
              className={`shrink-0 self-end mb-1.5 px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider ${
                mode === "analyst"
                  ? "bg-blue-900/60 text-blue-300 border border-blue-700/50"
                  : "bg-slate-700 text-slate-400 border border-slate-600"
              }`}
            >
              {mode}
            </span>

            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                mode === "analyst"
                  ? "Ask for a full analyst report (e.g. Analyze NVDA)…"
                  : "Ask a financial question (e.g. Latest AAPL news)…"
              }
              rows={1}
              disabled={loading}
              className="flex-1 bg-transparent resize-none outline-none text-sm text-slate-100 placeholder-slate-600 py-1.5 min-h-[36px] max-h-[120px] overflow-y-auto disabled:opacity-50"
            />

            <button
              onClick={() => sendMessage(input)}
              disabled={loading || !input.trim()}
              className="shrink-0 self-end mb-0.5 w-8 h-8 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white flex items-center justify-center transition-all disabled:cursor-not-allowed"
              aria-label="Send message"
            >
              {loading ? (
                <span className="w-3.5 h-3.5 border-2 border-slate-500 border-t-transparent rounded-full animate-spin" />
              ) : (
                <svg
                  className="w-3.5 h-3.5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5"
                  />
                </svg>
              )}
            </button>
          </div>

          <p className="text-center text-[10px] text-slate-700 mt-2">
            Press Enter to send · Shift+Enter for new line ·{" "}
            {mode === "analyst"
              ? "Analyst mode: detailed bull/bear report"
              : "Simple mode: concise Q&A with sources"}
          </p>
        </div>

        {/* Stats footer */}
        <StatsFooter stats={stats} />
      </div>
    </div>
  );
}
