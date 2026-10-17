// src/components/trade/TradeCloseDialog.tsx
import { useState, useEffect } from "react";
import { tradesApi } from "@/services/api";

const EMOTIONS = ["calm", "confident", "anxious", "satisfied", "frustrated", "relieved", "regretful"];

interface Props {
  tradeId: string;
  isOpen: boolean;
  onClose: () => void;
}

export function TradeCloseDialog({ tradeId, isOpen, onClose }: Props) {
  const [exitPrice, setExitPrice] = useState("");
  const [emotion, setEmotion] = useState<string | undefined>();
  const [notes, setNotes] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) {
      setExitPrice("");
      setEmotion(undefined);
      setNotes("");
      setError(null);
    }
  }, [isOpen]);

  const submit = async () => {
    if (!exitPrice) { setError("Exit price required"); return; }
    setLoading(true);
    try {
      await tradesApi.close(tradeId, {
        exit_price: parseFloat(exitPrice),
        emotion_post: emotion,
        exit_notes: notes,
        exit_time: new Date().toISOString(),
      });
      onClose();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-sm card p-6 animate-slide-up">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-bold text-[#F1F5F9]">Close Trade</h2>
          <button onClick={onClose} className="text-[#64748B] hover:text-[#94A3B8] text-lg">×</button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-[10px] uppercase tracking-wide text-[#64748B] mb-1 block">Exit Price (₹)</label>
            <input
              type="number"
              placeholder="0.00"
              value={exitPrice}
              onChange={(e) => setExitPrice(e.target.value)}
              className="w-full bg-[#0F1420] border border-[#2A3050] rounded-lg px-3 py-2 text-sm text-[#F1F5F9] focus:outline-none focus:border-[#8B5CF6] font-num"
              autoFocus
            />
          </div>

          <div>
            <label className="text-[10px] uppercase tracking-wide text-[#64748B] mb-1.5 block">Post-trade Emotion</label>
            <div className="flex flex-wrap gap-1.5">
              {EMOTIONS.map((em) => (
                <button
                  key={em}
                  onClick={() => setEmotion(emotion === em ? undefined : em)}
                  className={`px-2 py-1 rounded text-[11px] capitalize transition-all ${
                    emotion === em
                      ? "bg-[#8B5CF6]/20 text-[#A78BFA] border border-[#8B5CF6]/40"
                      : "bg-[#0F1420] border border-[#2A3050] text-[#64748B] hover:text-[#94A3B8]"
                  }`}
                >
                  {em}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-[10px] uppercase tracking-wide text-[#64748B] mb-1 block">Exit Notes</label>
            <textarea
              placeholder="Why did you exit?"
              rows={2}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="w-full bg-[#0F1420] border border-[#2A3050] rounded-lg px-3 py-2 text-sm text-[#F1F5F9] focus:outline-none focus:border-[#8B5CF6] resize-none"
            />
          </div>

          {error && (
            <div className="text-xs text-[#EF4444] bg-[#EF4444]/10 border border-[#EF4444]/20 rounded px-3 py-2">{error}</div>
          )}

          <button
            onClick={submit}
            disabled={loading}
            className="w-full py-2.5 rounded-lg ai-gradient text-white text-sm font-semibold hover:opacity-90 disabled:opacity-50 transition-all"
          >
            {loading ? "Closing..." : "Close Trade"}
          </button>
        </div>
      </div>
    </div>
  );
}
