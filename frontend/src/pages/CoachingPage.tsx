// src/pages/CoachingPage.tsx
import { useState, useRef, useEffect } from "react";
import { chatApi, type ChatMessageResponse, type AgentInsight } from "@/services/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  insights?: AgentInsight[];
  timestamp: Date;
}

const QUICK_QUESTIONS = [
  "How am I doing this week?",
  "What are my worst behavioral patterns?",
  "Which strategy should I focus on in current regime?",
  "Analyze my last 10 trades",
  "Am I ready to increase position size?",
  "What is my best time of day to trade?",
];

function ChatBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} animate-slide-up`}>
      <div className={`max-w-[80%] ${isUser ? "items-end" : "items-start"} flex flex-col gap-1`}>
        {!isUser && (
          <div className="flex items-center gap-1.5 mb-0.5">
            <div className="h-5 w-5 rounded ai-gradient flex items-center justify-center">
              <span className="text-[9px] text-white font-bold">✦</span>
            </div>
            <span className="text-[10px] text-[#64748B]">AI Coach</span>
          </div>
        )}
        <div
          className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
            isUser
              ? "bg-[#8B5CF6]/20 border border-[#8B5CF6]/30 text-[#F1F5F9] rounded-tr-sm"
              : "bg-[#1A2035] border border-[#2A3050] text-[#CBD5E1] rounded-tl-sm"
          }`}
        >
          {msg.content}
        </div>

        {/* Insights from AI */}
        {msg.insights && msg.insights.length > 0 && (
          <div className="space-y-1.5 mt-1">
            {msg.insights.slice(0, 2).map((ins, i) => {
              const color =
                ins.severity === "positive" ? "#10B981" :
                ins.severity === "warning" ? "#F59E0B" :
                ins.severity === "critical" ? "#EF4444" : "#3B82F6";
              return (
                <div
                  key={i}
                  className="px-3 py-2 rounded-lg border text-xs"
                  style={{ borderColor: color + "30", background: color + "08" }}
                >
                  <div className="font-semibold text-[#F1F5F9]">{ins.title}</div>
                  <div className="text-[#94A3B8] mt-0.5">{ins.message}</div>
                </div>
              );
            })}
          </div>
        )}

        <span className="text-[9px] text-[#3A4060]">
          {msg.timestamp.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })}
        </span>
      </div>
    </div>
  );
}

export default function CoachingPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Hello! I'm your AI trading coach. I have full context of your trading history, behavioral patterns, and current performance. What would you like to explore today?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async (text?: string) => {
    const msg = text || input.trim();
    if (!msg || loading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: msg,
      timestamp: new Date(),
    };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await chatApi.sendMessage({ message: msg, session_id: sessionId });
      setSessionId(res.session_id);
      const aiMsg: Message = {
        id: res.session_id + Date.now(),
        role: "assistant",
        content: res.message,
        insights: res.insights,
        timestamp: new Date(res.timestamp),
      };
      setMessages((m) => [...m, aiMsg]);
    } catch (e: any) {
      setMessages((m) => [
        ...m,
        {
          id: "err-" + Date.now(),
          role: "assistant",
          content: "Sorry, I couldn't connect to the coaching service. Please try again.",
          timestamp: new Date(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-56px)]">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[#2A3050] flex items-center gap-3 shrink-0">
        <div className="h-8 w-8 rounded-lg ai-gradient flex items-center justify-center shadow-glow-purple">
          <span className="text-white text-sm">✦</span>
        </div>
        <div>
          <h1 className="text-sm font-bold text-[#F1F5F9]">AI Coach</h1>
          <p className="text-[10px] text-[#64748B]">Personalized trading intelligence</p>
        </div>
        <div className="ml-auto flex items-center gap-1">
          <div className="h-2 w-2 rounded-full bg-[#10B981] pulse-dot" />
          <span className="text-[10px] text-[#10B981]">Online</span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((m) => (
          <ChatBubble key={m.id} msg={m} />
        ))}
        {loading && (
          <div className="flex justify-start animate-fade-in">
            <div className="bg-[#1A2035] border border-[#2A3050] rounded-2xl rounded-tl-sm px-4 py-3">
              <div className="flex gap-1 items-center">
                {[0, 1, 2].map((i) => (
                  <div
                    key={i}
                    className="h-1.5 w-1.5 rounded-full bg-[#8B5CF6] animate-bounce"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </div>
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Quick questions */}
      <div className="px-4 pb-2 flex gap-2 overflow-x-auto shrink-0">
        {QUICK_QUESTIONS.map((q) => (
          <button
            key={q}
            onClick={() => send(q)}
            className="shrink-0 px-3 py-1.5 rounded-full border border-[#2A3050] text-[11px] text-[#64748B] hover:border-[#8B5CF6]/40 hover:text-[#A78BFA] hover:bg-[#8B5CF6]/10 transition-all whitespace-nowrap"
          >
            {q}
          </button>
        ))}
      </div>

      {/* Input */}
      <div className="px-4 pb-4 pt-2 border-t border-[#2A3050] shrink-0">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Ask anything about your trading..."
            className="flex-1 bg-[#0F1420] border border-[#2A3050] rounded-xl px-4 py-2.5 text-sm text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6] transition-colors"
          />
          <button
            onClick={() => send()}
            disabled={!input.trim() || loading}
            className="px-4 py-2.5 rounded-xl ai-gradient text-white text-sm font-semibold hover:opacity-90 disabled:opacity-40 transition-all active:scale-95"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
