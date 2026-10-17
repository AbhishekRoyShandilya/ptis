// src/components/dashboard/ChatPanel.tsx
import { useState, useRef, useEffect, useCallback } from "react";
import { useCoPlannerStore, type ChatMessage } from "@/store/coPlannerStore";
import { useVoiceInput } from "@/hooks/useVoiceInput";

// ── Lightweight Markdown Renderer ───────────────────────────────────
// Handles: **bold**, *italic*, `code`, ## headings, --- dividers,
// - bullet lists, numbered lists, and line breaks. No external deps.

function renderInline(text: string): React.ReactNode[] {
  // Process **bold**, *italic*, `code` inline
  const parts: React.ReactNode[] = [];
  const re = /(\*\*(.+?)\*\*|\*(.+?)\*|`([^`]+)`)/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let key = 0;
  while ((match = re.exec(text)) !== null) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    if (match[2] !== undefined)
      parts.push(<strong key={key++} className="font-semibold text-[#F1F5F9]">{match[2]}</strong>);
    else if (match[3] !== undefined)
      parts.push(<em key={key++} className="italic text-[#CBD5E1]">{match[3]}</em>);
    else if (match[4] !== undefined)
      parts.push(
        <code key={key++} className="px-1 py-0.5 rounded bg-[#0D1117] text-[#7DD3FC] text-xs font-mono">
          {match[4]}
        </code>
      );
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

function MarkdownContent({ content }: { content: string }) {
  const lines = content.split("\n");
  const nodes: React.ReactNode[] = [];
  let key = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Horizontal rule
    if (/^---+$/.test(line.trim())) {
      nodes.push(<hr key={key++} className="border-[#2A3050] my-2" />);
      continue;
    }

    // ## heading
    if (line.startsWith("## ")) {
      nodes.push(
        <p key={key++} className="font-bold text-[#A78BFA] text-sm mt-3 mb-1">
          {renderInline(line.slice(3))}
        </p>
      );
      continue;
    }

    // # heading
    if (line.startsWith("# ")) {
      nodes.push(
        <p key={key++} className="font-bold text-[#C4B5FD] text-base mt-3 mb-1">
          {renderInline(line.slice(2))}
        </p>
      );
      continue;
    }

    // Bullet list (-, *, •)
    if (/^(\s*[-*•])\s+/.test(line)) {
      const indent = line.match(/^(\s*)/)?.[1].length ?? 0;
      const text = line.replace(/^\s*[-*•]\s+/, "");
      nodes.push(
        <div key={key++} className="flex gap-1.5 my-0.5" style={{ paddingLeft: indent * 4 }}>
          <span className="text-[#8B5CF6] mt-0.5 shrink-0">•</span>
          <span>{renderInline(text)}</span>
        </div>
      );
      continue;
    }

    // Numbered list
    const numMatch = line.match(/^(\s*)(\d+)\.\s+(.+)/);
    if (numMatch) {
      nodes.push(
        <div key={key++} className="flex gap-1.5 my-0.5">
          <span className="text-[#8B5CF6] font-mono text-xs mt-0.5 shrink-0">{numMatch[2]}.</span>
          <span>{renderInline(numMatch[3])}</span>
        </div>
      );
      continue;
    }

    // Empty line → small gap
    if (line.trim() === "") {
      nodes.push(<div key={key++} className="h-1.5" />);
      continue;
    }

    // Plain paragraph
    nodes.push(
      <p key={key++} className="leading-relaxed">
        {renderInline(line)}
      </p>
    );
  }

  return <div className="space-y-0.5 text-sm text-[#CBD5E1]">{nodes}</div>;
}

// ── Message Bubble ──────────────────────────────────────────────────

function MessageBubble({ msg }: { msg: ChatMessage }) {
  if (msg.role === "system") {
    // Short single-line system messages (alerts, status) → pill style
    // Multi-line system messages (battle plan, briefing) → card style
    const isMultiLine = msg.content.includes("\n");
    if (!isMultiLine) {
      return (
        <div className="flex justify-center animate-fade-in">
          <div className="px-3 py-1.5 rounded-full bg-[#8B5CF6]/10 border border-[#8B5CF6]/20 text-[10px] text-[#A78BFA]">
            {msg.content}
          </div>
        </div>
      );
    }
    return (
      <div className="flex justify-center animate-fade-in">
        <div className="w-full max-w-[90%] px-4 py-3 rounded-xl bg-[#8B5CF6]/5 border border-[#8B5CF6]/20 text-[#A78BFA]">
          <MarkdownContent content={msg.content} />
        </div>
      </div>
    );
  }

  const isUser = msg.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} animate-slide-up`}>
      <div className={`max-w-[75%] flex flex-col ${isUser ? "items-end" : "items-start"} gap-1`}>
        {/* AI avatar */}
        {!isUser && (
          <div className="flex items-center gap-1.5 mb-0.5">
            <div className="h-5 w-5 rounded ai-gradient flex items-center justify-center">
              <span className="text-[9px] text-white font-bold">✦</span>
            </div>
            <span className="text-[10px] text-[#64748B]">Co-Planner</span>
          </div>
        )}

        {/* Message body */}
        <div
          className={`px-4 py-3 rounded-2xl ${
            isUser
              ? "bg-[#8B5CF6]/20 border border-[#8B5CF6]/30 text-[#F1F5F9] rounded-tr-sm"
              : "bg-[#1A2035] border border-[#2A3050] text-[#CBD5E1] rounded-tl-sm"
          }`}
        >
          {isUser
            ? <span className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</span>
            : <MarkdownContent content={msg.content} />
          }
        </div>

        {/* Attachments */}
        {msg.attachments && msg.attachments.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-1">
            {msg.attachments.map((att, i) => (
              <div
                key={i}
                className="flex items-center gap-1 px-2 py-1 rounded bg-[#1A2035] border border-[#2A3050] text-[10px] text-[#94A3B8]"
              >
                <span>{att.type.startsWith("image/") ? "🖼" : "📎"}</span>
                <span className="truncate max-w-[100px]">{att.name}</span>
              </div>
            ))}
          </div>
        )}

        {/* AI insights */}
        {msg.insights && msg.insights.length > 0 && (
          <div className="space-y-1.5 mt-1 w-full">
            {msg.insights.slice(0, 3).map((ins, i) => {
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

        {/* Action items */}
        {msg.actionItems && msg.actionItems.length > 0 && (
          <div className="mt-1 px-3 py-2 rounded-lg bg-[#10B981]/5 border border-[#10B981]/20 text-xs w-full">
            <div className="font-semibold text-[#10B981] mb-1">Action Items</div>
            {msg.actionItems.map((item, i) => (
              <div key={i} className="text-[#94A3B8] flex items-start gap-1.5">
                <span className="text-[#10B981] mt-0.5">•</span>
                {item}
              </div>
            ))}
          </div>
        )}

        {/* Timestamp */}
        <span className="text-[9px] text-[#3A4060]">
          {msg.timestamp.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })}
        </span>
      </div>
    </div>
  );
}

// ── Follow-up Pills ─────────────────────────────────────────────────

function FollowUpPills({ questions, onSelect }: {
  questions: string[];
  onSelect: (q: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1.5 px-4 pb-2">
      {questions.map((q, i) => (
        <button
          key={i}
          onClick={() => onSelect(q)}
          className="px-3 py-1.5 rounded-full text-[11px] bg-[#8B5CF6]/10 border border-[#8B5CF6]/20 text-[#A78BFA] hover:bg-[#8B5CF6]/20 transition-colors"
        >
          {q}
        </button>
      ))}
    </div>
  );
}

// ── Main Chat Panel ─────────────────────────────────────────────────

export default function ChatPanel() {
  const {
    messages,
    activeSymbol,
    chatLoading,
    attachments,
    briefingLoading,
    sendMessage,
    addAttachment,
    removeAttachment,
    clearChat,
    setActiveSymbol,
    prepareSession,
    startProactiveAlerts,
    stopProactiveAlerts,
  } = useCoPlannerStore();

  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imgInputRef = useRef<HTMLInputElement>(null);
  const { isListening, transcript, startListening, stopListening, isSupported: voiceSupported } =
    useVoiceInput();

  // Connect the proactive partner feed (ws/session): money moves building on the watchlist and
  // open-position story-changes arrive as chat bubbles — the partner speaks without being asked.
  useEffect(() => {
    const token = localStorage.getItem("ptis_access_token");
    if (token) startProactiveAlerts(token);
    return () => stopProactiveAlerts();
  }, [startProactiveAlerts, stopProactiveAlerts]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Voice transcript → input
  useEffect(() => {
    if (transcript) {
      setInput((prev) => (prev ? prev + " " : "") + transcript);
    }
  }, [transcript]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + "px";
    }
  }, [input]);

  const handleSend = useCallback(() => {
    if (!input.trim() && attachments.length === 0) return;
    sendMessage(input);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }, [input, attachments, sendMessage]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFilePick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      Array.from(files).forEach((f) => addAttachment(f));
    }
    e.target.value = "";
  };

  // Get follow-up questions from last AI message
  const lastAiMsg = [...messages].reverse().find((m) => m.role === "assistant");
  const followUps = lastAiMsg?.followUpQuestions ?? [];

  return (
    <div className="flex flex-col h-full bg-[#0A0E17]">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#2A3050]">
        <div className="flex items-center gap-3">
          <div className="h-7 w-7 rounded-lg ai-gradient flex items-center justify-center">
            <span className="text-xs text-white font-bold">✦</span>
          </div>
          <div>
            <h2 className="text-sm font-semibold text-[#F1F5F9]">Trade Co-Planner</h2>
            <p className="text-[10px] text-[#64748B]">AI-powered trade planning assistant</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Active stock badge */}
          {activeSymbol && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#8B5CF6]/10 border border-[#8B5CF6]/30">
              <div className="h-1.5 w-1.5 rounded-full bg-[#8B5CF6] animate-pulse" />
              <span className="text-xs font-semibold text-[#A78BFA] font-mono">
                {activeSymbol}
              </span>
              <button
                onClick={() => setActiveSymbol(null)}
                className="text-[#64748B] hover:text-[#F1F5F9] ml-0.5 text-xs"
              >
                ×
              </button>
            </div>
          )}
          <button
            onClick={() => prepareSession()}
            disabled={briefingLoading || chatLoading}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-medium transition-all ${
              briefingLoading
                ? "bg-[#8B5CF6]/10 text-[#A78BFA] cursor-wait"
                : "bg-[#8B5CF6]/15 border border-[#8B5CF6]/30 text-[#A78BFA] hover:bg-[#8B5CF6]/25"
            }`}
            title="Run pre-session analysis: watchlist evaluation, strategy fit, battle plan"
          >
            {briefingLoading ? (
              <>
                <div className="h-3 w-3 rounded-full border border-[#A78BFA] border-t-transparent animate-spin" />
                Preparing...
              </>
            ) : (
              <>
                <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                Prepare Session
              </>
            )}
          </button>
          <button
            onClick={clearChat}
            className="text-[10px] text-[#64748B] hover:text-[#94A3B8] px-2 py-1 rounded transition-colors"
            title="New chat"
          >
            New Chat
          </button>
        </div>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} msg={msg} />
        ))}

        {/* Typing indicator */}
        {chatLoading && (
          <div className="flex items-center gap-2 animate-fade-in">
            <div className="h-5 w-5 rounded ai-gradient flex items-center justify-center">
              <span className="text-[9px] text-white font-bold">✦</span>
            </div>
            <div className="flex gap-1">
              <div className="h-2 w-2 rounded-full bg-[#8B5CF6] animate-bounce" style={{ animationDelay: "0ms" }} />
              <div className="h-2 w-2 rounded-full bg-[#8B5CF6] animate-bounce" style={{ animationDelay: "150ms" }} />
              <div className="h-2 w-2 rounded-full bg-[#8B5CF6] animate-bounce" style={{ animationDelay: "300ms" }} />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Follow-up suggestions */}
      {followUps.length > 0 && !chatLoading && (
        <FollowUpPills
          questions={followUps}
          onSelect={(q) => {
            setInput(q);
            textareaRef.current?.focus();
          }}
        />
      )}

      {/* Attachment preview */}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-2 px-4 pb-2">
          {attachments.map((f, i) => (
            <div
              key={i}
              className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-[#1A2035] border border-[#2A3050] text-xs text-[#94A3B8]"
            >
              <span>{f.type.startsWith("image/") ? "🖼" : "📎"}</span>
              <span className="truncate max-w-[120px]">{f.name}</span>
              <button
                onClick={() => removeAttachment(i)}
                className="text-[#64748B] hover:text-[#EF4444] ml-1"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Input area */}
      <div className="px-4 pb-4 pt-2 border-t border-[#2A3050]">
        <div className="flex items-end gap-2 bg-[#1A2035] border border-[#2A3050] rounded-xl px-3 py-2 focus-within:border-[#8B5CF6]/50 transition-colors">
          {/* File attachment */}
          <button
            onClick={() => fileInputRef.current?.click()}
            className="p-1.5 text-[#64748B] hover:text-[#94A3B8] transition-colors shrink-0"
            title="Attach file"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
            </svg>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.csv,.txt,.json"
            className="hidden"
            onChange={handleFilePick}
          />

          {/* Image attachment */}
          <button
            onClick={() => imgInputRef.current?.click()}
            className="p-1.5 text-[#64748B] hover:text-[#94A3B8] transition-colors shrink-0"
            title="Attach screenshot"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <polyline points="21 15 16 10 5 21" />
            </svg>
          </button>
          <input
            ref={imgInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFilePick}
          />

          {/* Text input */}
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              activeSymbol
                ? `Ask about ${activeSymbol}...`
                : "Plan a trade, ask for analysis..."
            }
            rows={1}
            className="flex-1 bg-transparent text-sm text-[#F1F5F9] placeholder-[#3A4060] resize-none focus:outline-none py-1"
            style={{ maxHeight: 120 }}
          />

          {/* Voice input */}
          {voiceSupported && (
            <button
              onClick={isListening ? stopListening : startListening}
              className={`p-1.5 transition-colors shrink-0 ${
                isListening
                  ? "text-[#EF4444] animate-pulse"
                  : "text-[#64748B] hover:text-[#94A3B8]"
              }`}
              title={isListening ? "Stop recording" : "Voice input"}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
                <path d="M19 10v2a7 7 0 01-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
            </button>
          )}

          {/* Send */}
          <button
            onClick={handleSend}
            disabled={chatLoading || (!input.trim() && attachments.length === 0)}
            className={`p-1.5 rounded-lg transition-all shrink-0 ${
              input.trim() || attachments.length > 0
                ? "bg-[#8B5CF6] text-white hover:bg-[#7C3AED]"
                : "text-[#3A4060] cursor-not-allowed"
            }`}
            title="Send message"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>

        <div className="text-[10px] text-[#3A4060] text-center mt-2">
          Shift+Enter for new line • AI responses are for informational purposes only
        </div>
      </div>
    </div>
  );
}
