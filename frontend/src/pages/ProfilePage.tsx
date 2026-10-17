// src/pages/ProfilePage.tsx
import { useEffect, useState } from "react";
import { profileApi } from "@/services/api";

const STAT_LABEL: Record<string, string> = {
  win_rate: "Win Rate",
  avg_r: "Avg R-Multiple",
  profit_factor: "Profit Factor",
  total_trades: "Total Trades",
  avg_discipline_score: "Avg Discipline",
  best_entry_emotion: "Best Emotion",
  worst_entry_emotion: "Worst Emotion",
  dominant_regime: "Best Regime",
};

function ScoreRing({ value, label }: { value: number; label: string }) {
  const pct = Math.min(Math.max(value, 0), 100);
  const colour = pct >= 70 ? "#10B981" : pct >= 40 ? "#F59E0B" : "#EF4444";
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative w-16 h-16">
        <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
          <circle cx="18" cy="18" r="15.9" fill="none" stroke="#1E293B" strokeWidth="3" />
          <circle
            cx="18" cy="18" r="15.9" fill="none"
            stroke={colour} strokeWidth="3"
            strokeDasharray={`${pct} ${100 - pct}`}
            strokeLinecap="round"
          />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-[#F1F5F9]">
          {Math.round(pct)}
        </span>
      </div>
      <span className="text-[10px] text-[#64748B] text-center leading-tight">{label}</span>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: any }) {
  const display = typeof value === "number"
    ? (label.includes("Rate") ? `${(value * 100).toFixed(1)}%`
      : label.includes("R-") ? value.toFixed(2)
      : label.includes("Factor") ? value.toFixed(2)
      : Math.round(value).toString())
    : String(value ?? "—");

  return (
    <div className="bg-[#1E293B] rounded-xl p-3 flex flex-col gap-1">
      <span className="text-[10px] text-[#64748B] uppercase tracking-wider">{label}</span>
      <span className="text-lg font-bold text-[#F1F5F9]">{display}</span>
    </div>
  );
}

export default function ProfilePage() {
  const [dna, setDna] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    profileApi.dna()
      .then(setDna)
      .catch(() => setDna(null))
      .finally(() => setLoading(false));
  }, []);

  // Real data only — NO demo/sample fallback. A fresh user with no trades
  // sees an honest empty state, not fabricated numbers.
  const hasData = !!dna && !dna.error && (dna.total_trades ?? 0) > 0;

  if (loading) {
    return (
      <div className="p-4 max-w-[900px] mx-auto">
        <div className="text-[#64748B] text-sm animate-pulse">Loading profile…</div>
      </div>
    );
  }

  if (!hasData) {
    return (
      <div className="p-4 max-w-[900px] mx-auto animate-fade-in">
        <div className="mb-5">
          <h1 className="text-xl font-bold text-[#F1F5F9]">Trader Profile</h1>
          <p className="text-xs text-[#64748B] mt-0.5">Your personal trading DNA — behavioural &amp; performance identity</p>
        </div>
        <div className="bg-[#152035] rounded-2xl p-8 text-center">
          <div className="text-3xl mb-3">🧬</div>
          <h2 className="text-sm font-semibold text-[#F1F5F9] mb-1">No profile yet</h2>
          <p className="text-xs text-[#64748B] max-w-md mx-auto leading-relaxed">
            Your Trader Profile is built entirely from <span className="text-[#94A3B8]">your</span> trades.
            Connect your broker and sync your history (Settings → Broker), or it
            grows automatically as you log trades. Discipline, behavioural
            patterns, and edge appear once you have real trades.
          </p>
        </div>
      </div>
    );
  }

  const data = dna;

  return (
    <div className="p-4 max-w-[900px] mx-auto animate-fade-in">
      {/* Header */}
      <div className="mb-5">
        <h1 className="text-xl font-bold text-[#F1F5F9]">Trader Profile</h1>
        <p className="text-xs text-[#64748B] mt-0.5">Your personal trading DNA — behavioural & performance identity</p>
      </div>

      {/* Score rings */}
      <div className="bg-[#152035] rounded-2xl p-5 mb-4 flex flex-wrap gap-6 justify-around">
        <ScoreRing value={(data.win_rate || 0) * 100} label="Win Rate" />
        <ScoreRing value={(data.avg_discipline_score || 0)} label="Discipline" />
        <ScoreRing value={Math.min((data.profit_factor || 0) * 33, 100)} label="Profit Factor" />
        <ScoreRing value={Math.min((data.avg_r || 0) * 35, 100)} label="Avg R" />
      </div>

      {/* Key stats grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        {Object.entries(STAT_LABEL).map(([key, label]) => (
          <StatCard key={key} label={label} value={data[key]} />
        ))}
      </div>

      {/* Behavioural archetypes */}
      {data.behavioural_archetypes && (
        <div className="bg-[#1E293B] rounded-xl p-4 mb-4">
          <h2 className="text-sm font-semibold text-[#F1F5F9] mb-3">Behavioural Archetypes</h2>
          <div className="space-y-2">
            {data.behavioural_archetypes.map((a: any) => (
              <div key={a.label}>
                <div className="flex justify-between text-xs text-[#94A3B8] mb-1">
                  <span>{a.label}</span>
                  <span>{a.pct}%</span>
                </div>
                <div className="w-full bg-[#0F172A] rounded-full h-1.5">
                  <div
                    className="h-1.5 rounded-full bg-[#3B82F6]"
                    style={{ width: `${a.pct}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Strengths & Growth */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {data.strengths && (
          <div className="bg-[#1E293B] rounded-xl p-4">
            <h2 className="text-sm font-semibold text-[#10B981] mb-3">Strengths</h2>
            <ul className="space-y-1.5">
              {data.strengths.map((s: string, i: number) => (
                <li key={i} className="flex gap-2 text-xs text-[#94A3B8]">
                  <span className="text-[#10B981] mt-0.5">✓</span>
                  {s}
                </li>
              ))}
            </ul>
          </div>
        )}
        {data.growth_areas && (
          <div className="bg-[#1E293B] rounded-xl p-4">
            <h2 className="text-sm font-semibold text-[#F59E0B] mb-3">Growth Areas</h2>
            <ul className="space-y-1.5">
              {data.growth_areas.map((s: string, i: number) => (
                <li key={i} className="flex gap-2 text-xs text-[#94A3B8]">
                  <span className="text-[#F59E0B] mt-0.5">→</span>
                  {s}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {loading && (
        <div className="fixed inset-0 flex items-center justify-center bg-black/30 z-50">
          <div className="text-[#64748B] text-sm animate-pulse">Loading profile…</div>
        </div>
      )}
    </div>
  );
}
