// src/components/trade/TradeEntryDialog.tsx
import { useState, useEffect } from "react";
import { tradesApi, type TradeEntryRequest, type AgentInsight } from "@/services/api";

const EMOTIONS = ["calm", "confident", "anxious", "excited", "fearful", "frustrated", "euphoric"];

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export function TradeEntryDialog({ isOpen, onClose }: Props) {
  const [form, setForm] = useState<TradeEntryRequest>({
    symbol: "",
    direction: "long",
    entry_price: 0,
    position_size: 1,
    source: "manual",
  });
  const [loading, setLoading] = useState(false);
  const [insights, setInsights] = useState<AgentInsight[]>([]);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) {
      setForm({ symbol: "", direction: "long", entry_price: 0, position_size: 1, source: "manual" });
      setInsights([]);
      setSubmitted(false);
      setError(null);
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

  const submit = async () => {
    if (!form.symbol || !form.entry_price) {
      setError("Symbol and entry price are required");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await tradesApi.enter({
        ...form,
        entry_time: new Date().toISOString(),
      });
      setInsights(res.insights);
      setSubmitted(true);
    } catch (e: any) {
      setError(e.message || "Failed to log trade");
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-md card p-6 animate-slide-up shadow-card-hover">
        {!submitted ? (
          <>
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-base font-bold text-[#F1F5F9]">Log Trade</h2>
              <button
                onClick={onClose}
                className="text-[#64748B] hover:text-[#94A3B8] text-lg leading-none"
              >
                ×
              </button>
            </div>

            <div className="space-y-4">
              {/* Symbol + Direction */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-[10px] uppercase tracking-wide text-[#64748B] mb-1 block">
                    Symbol
                  </label>
                  <input
                    type="text"
                    placeholder="NIFTY, RELIANCE..."
                    value={form.symbol}
                    onChange={(e) =>
                      setForm((f) => ({ ...f, symbol: e.target.value.toUpperCase() }))
                    }
                    className="w-full bg-[#0F1420] border border-[#2A3050] rounded-lg px-3 py-2 text-sm text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6] transition-colors font-num"
                  />
                </div>
                <div>
                  <label className="text-[10px] uppercase tracking-wide text-[#64748B] mb-1 block">
                    Direction
                  </label>
                  <div className="flex gap-1">
                    {["long", "short"].map((d) => (
                      <button
                        key={d}
                        onClick={() => setForm((f) => ({ ...f, direction: d }))}
                        className={`flex-1 py-2 rounded-lg text-xs font-semibold uppercase transition-all ${
                          form.direction === d
                            ? d === "long"
                              ? "bg-[#10B981]/20 text-[#10B981] border border-[#10B981]/40"
                              : "bg-[#EF4444]/20 text-[#EF4444] border border-[#EF4444]/40"
                            : "bg-[#0F1420] border border-[#2A3050] text-[#64748B]"
                        }`}
                      >
                        {d}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Price + Size */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-[10px] uppercase tracking-wide text-[#64748B] mb-1 block">
                    Entry Price (₹)
                  </label>
                  <input
                    type="number"
                    placeholder="0.00"
                    value={form.entry_price || ""}
                    onChange={(e) =>
                      setForm((f) => ({ ...f, entry_price: parseFloat(e.target.value) || 0 }))
                    }
                    className="w-full bg-[#0F1420] border border-[#2A3050] rounded-lg px-3 py-2 text-sm text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6] transition-colors font-num"
                  />
                </div>
                <div>
                  <label className="text-[10px] uppercase tracking-wide text-[#64748B] mb-1 block">
                    Position Size
                  </label>
                  <input
                    type="number"
                    placeholder="Qty / Lots"
                    value={form.position_size || ""}
                    onChange={(e) =>
                      setForm((f) => ({ ...f, position_size: parseFloat(e.target.value) || 0 }))
                    }
                    className="w-full bg-[#0F1420] border border-[#2A3050] rounded-lg px-3 py-2 text-sm text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6] transition-colors font-num"
                  />
                </div>
              </div>

              {/* Stop / Target */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-[10px] uppercase tracking-wide text-[#64748B] mb-1 block">
                    Stop Loss (₹)
                  </label>
                  <input
                    type="number"
                    placeholder="Optional"
                    onChange={(e) =>
                      setForm((f) => ({ ...f, stop_loss: parseFloat(e.target.value) || undefined }))
                    }
                    className="w-full bg-[#0F1420] border border-[#2A3050] rounded-lg px-3 py-2 text-sm text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6] transition-colors font-num"
                  />
                </div>
                <div>
                  <label className="text-[10px] uppercase tracking-wide text-[#64748B] mb-1 block">
                    Take Profit (₹)
                  </label>
                  <input
                    type="number"
                    placeholder="Optional"
                    onChange={(e) =>
                      setForm((f) => ({ ...f, take_profit: parseFloat(e.target.value) || undefined }))
                    }
                    className="w-full bg-[#0F1420] border border-[#2A3050] rounded-lg px-3 py-2 text-sm text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6] transition-colors font-num"
                  />
                </div>
              </div>

              {/* Strategy */}
              <div>
                <label className="text-[10px] uppercase tracking-wide text-[#64748B] mb-1 block">
                  Strategy (short code)
                </label>
                <input
                  type="text"
                  placeholder="e.g. CMPBR01, ORB, VWAP..."
                  value={form.declared_strategy_id || ""}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, declared_strategy_id: e.target.value.toUpperCase() || undefined }))
                  }
                  className="w-full bg-[#0F1420] border border-[#2A3050] rounded-lg px-3 py-2 text-sm text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6] transition-colors font-num"
                />
              </div>

              {/* Emotion */}
              <div>
                <label className="text-[10px] uppercase tracking-wide text-[#64748B] mb-1.5 block">
                  Pre-trade Emotion
                </label>
                <div className="flex flex-wrap gap-1.5">
                  {EMOTIONS.map((em) => (
                    <button
                      key={em}
                      onClick={() =>
                        setForm((f) => ({
                          ...f,
                          emotion_pre: f.emotion_pre === em ? undefined : em,
                        }))
                      }
                      className={`px-2 py-1 rounded text-[11px] capitalize transition-all ${
                        form.emotion_pre === em
                          ? "bg-[#8B5CF6]/20 text-[#A78BFA] border border-[#8B5CF6]/40"
                          : "bg-[#0F1420] border border-[#2A3050] text-[#64748B] hover:text-[#94A3B8]"
                      }`}
                    >
                      {em}
                    </button>
                  ))}
                </div>
              </div>

              {/* Conviction */}
              <div>
                <label className="text-[10px] uppercase tracking-wide text-[#64748B] mb-1.5 block">
                  Conviction {form.conviction_score ? `${form.conviction_score}/10` : ""}
                </label>
                <input
                  type="range"
                  min={1}
                  max={10}
                  step={1}
                  value={form.conviction_score ?? 5}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, conviction_score: parseInt(e.target.value) }))
                  }
                  className="w-full accent-[#8B5CF6]"
                />
              </div>

              {/* Notes */}
              <div>
                <label className="text-[10px] uppercase tracking-wide text-[#64748B] mb-1 block">
                  Notes
                </label>
                <textarea
                  placeholder="Setup notes, thesis..."
                  rows={2}
                  onChange={(e) => setForm((f) => ({ ...f, notes: e.target.value }))}
                  className="w-full bg-[#0F1420] border border-[#2A3050] rounded-lg px-3 py-2 text-sm text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6] transition-colors resize-none"
                />
              </div>

              {error && (
                <div className="text-xs text-[#EF4444] bg-[#EF4444]/10 border border-[#EF4444]/20 rounded px-3 py-2">
                  {error}
                </div>
              )}

              <button
                onClick={submit}
                disabled={loading}
                className="w-full py-2.5 rounded-lg ai-gradient text-white text-sm font-semibold hover:opacity-90 disabled:opacity-50 transition-all active:scale-[0.98]"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="h-4 w-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Processing...
                  </span>
                ) : (
                  "Log Trade +"
                )}
              </button>
            </div>
          </>
        ) : (
          /* Success State */
          <div className="text-center animate-fade-in">
            <div className="h-12 w-12 rounded-full bg-[#10B981]/20 border border-[#10B981]/40 flex items-center justify-center mx-auto mb-4">
              <span className="text-[#10B981] text-xl">✓</span>
            </div>
            <h3 className="text-base font-bold text-[#F1F5F9] mb-1">Trade Logged</h3>
            <p className="text-xs text-[#64748B] mb-4">
              AI attribution & shadow trades created
            </p>

            {insights.length > 0 && (
              <div className="space-y-2 mb-4 text-left">
                {insights.slice(0, 3).map((ins, i) => {
                  const borderColor =
                    ins.severity === "positive"
                      ? "#10B981"
                      : ins.severity === "warning"
                      ? "#F59E0B"
                      : ins.severity === "critical"
                      ? "#EF4444"
                      : "#3B82F6";
                  return (
                    <div
                      key={i}
                      className="p-2.5 rounded-lg border text-left"
                      style={{ borderColor: borderColor + "30", background: borderColor + "08" }}
                    >
                      <div className="text-xs font-semibold text-[#F1F5F9]">{ins.title}</div>
                      <div className="text-[11px] text-[#94A3B8] mt-0.5">{ins.message}</div>
                    </div>
                  );
                })}
              </div>
            )}

            <div className="flex gap-2">
              <button
                onClick={() => { setSubmitted(false); setInsights([]); setForm({ symbol: "", direction: "long", entry_price: 0, position_size: 1, source: "manual" }); }}
                className="flex-1 py-2 rounded-lg border border-[#2A3050] text-xs text-[#94A3B8] hover:bg-[#1E2540] transition-colors"
              >
                Log Another
              </button>
              <button
                onClick={onClose}
                className="flex-1 py-2 rounded-lg ai-gradient text-white text-xs font-semibold hover:opacity-90"
              >
                Done
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
