import type {
  ChatRequest,
  SimpleAnswer,
  AnalystAnswer,
  StatsResponse,
  HealthResponse,
} from "@/types/rag";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchJSON<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new Error(`API error ${res.status}: ${text}`);
  }

  return res.json() as Promise<T>;
}

export async function sendChat(
  request: ChatRequest
): Promise<SimpleAnswer | AnalystAnswer> {
  return fetchJSON<SimpleAnswer | AnalystAnswer>("/api/chat", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function getStats(): Promise<StatsResponse> {
  return fetchJSON<StatsResponse>("/api/stats");
}

export async function getHealth(): Promise<HealthResponse> {
  return fetchJSON<HealthResponse>("/api/health");
}
