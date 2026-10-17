// src/components/journal/JournalDialog.tsx
import { useState, useEffect } from "react";
import { journalApi } from "@/services/api";

const MOODS = [
  { label: "😊 Confident", value: "confident" },
  { label: "😐 Neutral", value: "neutral" },
  { label: "😰 Anxious", value: "anxious" },
  { label: "😤 Frustrated", value: "frustrated" },
  { label: "🤩 Excited", value: "excited" },
  { label: "😔 Low", value: "low" },
];

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export function JournalDialog({ isOpen, onClose }: Props) {
  const [mode, setMode] = useState<"quick" | "voice">("quick");
  const [mood, setMood] = useState<string | null>(null);
  const [content, setContent] = useState("");
  const [symbol, setSymbol] = useState("");
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);
  const [recordingState, setRecordingState] = useState<"idle" | "recording" | "processing">("idle");
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (!isOpen) {
      setMood(null);
      setContent("");
      setSymbol("");
      setDone(false);
      setRecordingState("idle");
      setDuration(0);
    }
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isOpen, onClose]);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;
    if (recordingState === "recording") {
      timer = setInterval(() => setDuration((d) => d + 1), 1000);
    }
    return () => clearInterval(timer);
  }, [recordingState]);

  const handleQuickSubmit = async () => {
    setLoading(true);
    try {
      await journalApi.create({
        content,
        mood,
        symbol,
        timestamp: new Date().toISOString(),
        type: "quick",
      });
      setDone(true);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const startRecording = () => {
    setRecordingState("recording");
    setDuration(0);
  };

  const stopRecording = async () => {
    setRecordingState("processing");
    await new Promise((r) => setTimeout(r, 1500));
    setRecordingState("idle");
    setDone(true);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-md card p-5 animate-slide-up rounded-2xl">
        {done ? (
          <div className="text-center py-4 animate-fade-in">
            <div className="text-3xl mb-3">✅</div>
            <h3 className="text-base font-bold text-[#F1F5F9] mb-1">Journaled!</h3>
            <p className="text-xs text-[#64748B] mb-4">Your entry has been saved and analyzed.</p>
            <button
              onClick={onClose}
              className="px-6 py-2 rounded-lg ai-gradient text-white text-sm font-semibold"
            >
              Done
            </button>
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <span className="text-base">◇</span>
                <h2 className="text-sm font-bold text-[#F1F5F9]">Quick Journal</h2>
              </div>
              <div className="flex items-center gap-2">
                {/* Mode toggle */}
                <div className="flex bg-[#0F1420] rounded-lg p-0.5">
                  {(["quick", "voice"] as const).map((m) => (
                    <button
                      key={m}
                      onClick={() => setMode(m)}
                      className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-all ${
                        mode === m
                          ? "bg-[#8B5CF6]/20 text-[#A78BFA]"
                          : "text-[#64748B]"
                      }`}
                    >
                      {m === "quick" ? "⌨️ Quick" : "🎤 Voice"}
                    </button>
                  ))}
                </div>
                <button onClick={onClose} className="text-[#64748B] hover:text-[#94A3B8] text-lg">×</button>
              </div>
            </div>

            {mode === "quick" ? (
              <div className="space-y-3">
                {/* Mood */}
                <div>
                  <label className="text-[10px] uppercase tracking-wide text-[#64748B] mb-1.5 block">
                    How are you feeling?
                  </label>
                  <div className="grid grid-cols-3 gap-1.5">
                    {MOODS.map((m) => (
                      <button
                        key={m.value}
                        onClick={() => setMood(mood === m.value ? null : m.value)}
                        className={`py-1.5 px-2 rounded-lg text-xs transition-all ${
                          mood === m.value
                            ? "bg-[#8B5CF6]/20 text-[#A78BFA] border border-[#8B5CF6]/40"
                            : "bg-[#0F1420] border border-[#2A3050] text-[#94A3B8] hover:border-[#3A4060]"
                        }`}
                      >
                        {m.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Symbol */}
                <div>
                  <input
                    type="text"
                    placeholder="Symbol (optional)"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                    className="w-full bg-[#0F1420] border border-[#2A3050] rounded-lg px-3 py-2 text-sm text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6] font-num"
                  />
                </div>

                {/* Content */}
                <textarea
                  placeholder="What's on your mind? Market conditions, setup, lessons..."
                  rows={4}
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  className="w-full bg-[#0F1420] border border-[#2A3050] rounded-lg px-3 py-2 text-sm text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6] resize-none"
                />

                <button
                  onClick={handleQuickSubmit}
                  disabled={loading || (!content && !mood)}
                  className="w-full py-2.5 rounded-lg ai-gradient text-white text-sm font-semibold hover:opacity-90 disabled:opacity-40 transition-all"
                >
                  {loading ? "Saving..." : "Save Journal Entry"}
                </button>
              </div>
            ) : (
              /* Voice mode */
              <div className="text-center py-4 space-y-5">
                {recordingState === "idle" && (
                  <>
                    <p className="text-xs text-[#94A3B8]">
                      Speak your trade thoughts. AI will extract key details automatically.
                    </p>
                    <button
                      onClick={startRecording}
                      className="h-20 w-20 rounded-full bg-[#8B5CF6]/20 border-2 border-[#8B5CF6]/40 flex items-center justify-center mx-auto text-3xl hover:bg-[#8B5CF6]/30 transition-all"
                    >
                      🎤
                    </button>
                    <p className="text-xs text-[#64748B]">Tap to start recording</p>
                  </>
                )}
                {recordingState === "recording" && (
                  <>
                    <div className="flex items-center justify-center gap-1 mb-2">
                      {[...Array(5)].map((_, i) => (
                        <div
                          key={i}
                          className="w-1 rounded-full bg-[#EF4444] animate-pulse"
                          style={{
                            height: `${12 + Math.random() * 20}px`,
                            animationDelay: `${i * 0.1}s`,
                          }}
                        />
                      ))}
                    </div>
                    <div className="text-sm font-num text-[#EF4444]">
                      {String(Math.floor(duration / 60)).padStart(2, "0")}:{String(duration % 60).padStart(2, "0")}
                    </div>
                    <button
                      onClick={stopRecording}
                      className="h-16 w-16 rounded-full bg-[#EF4444]/20 border-2 border-[#EF4444]/40 flex items-center justify-center mx-auto text-2xl hover:bg-[#EF4444]/30 transition-all"
                    >
                      ⏹
                    </button>
                  </>
                )}
                {recordingState === "processing" && (
                  <div className="py-4">
                    <div className="h-6 w-6 border-2 border-[#8B5CF6] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                    <p className="text-xs text-[#94A3B8]">AI is extracting details...</p>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
