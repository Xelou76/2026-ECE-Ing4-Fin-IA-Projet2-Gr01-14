"use client";

import type { Message, RAGAnswer } from "@/types/rag";
import SimpleMessage from "./SimpleMessage";
import AnalystCard from "./AnalystCard";

interface ChatMessageProps {
  message: Message;
}

function isRAGAnswer(content: RAGAnswer | string): content is RAGAnswer {
  return typeof content === "object" && content !== null && "_mode" in content;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[80%] sm:max-w-[65%]">
          <div className="bg-blue-600 text-white rounded-2xl rounded-br-sm px-4 py-3 shadow-lg">
            <p className="text-sm leading-relaxed">
              {typeof message.content === "string" ? message.content : ""}
            </p>
          </div>
          <p className="text-right text-[10px] text-slate-600 mt-1 pr-1">
            {message.timestamp.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </p>
        </div>
      </div>
    );
  }

  // Assistant message
  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-[92%] sm:max-w-[80%]">
        {/* Avatar */}
        <div className="flex items-center gap-2 mb-2">
          <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center text-xs font-bold text-white shrink-0">
            BV
          </div>
          <span className="text-xs font-semibold text-slate-400">
            BrightVest AI
          </span>
          <span className="text-[10px] text-slate-600">
            {message.timestamp.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </span>
        </div>

        {/* Content card */}
        <div className="bg-slate-800 border border-slate-700/60 rounded-2xl rounded-tl-sm px-4 py-4 shadow-lg">
          {isRAGAnswer(message.content) ? (
            message.content._mode === "analyst" ? (
              <AnalystCard data={message.content} />
            ) : (
              <SimpleMessage data={message.content} />
            )
          ) : (
            // Error or plain string
            <p className="text-red-400 text-sm leading-relaxed">
              {String(message.content)}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
