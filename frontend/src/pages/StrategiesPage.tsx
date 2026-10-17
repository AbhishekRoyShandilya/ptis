// src/pages/StrategiesPage.tsx
import { useEffect, useState } from "react";
import { strategiesApi } from "@/services/api";

export default function StrategiesPage() {
  const [strategies, setStrategies] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    strategiesApi.list().then((d: any) => setStrategies(Array.isArray(d) ? d : [])).catch(() => {}).finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="p-4 max-w-[900px] mx-auto">
        <div className="text-[#64748B] text-sm animate-pulse">Loading strategies…</div>
      </div>
    );
  }

  // No demo/sample strategies. A fresh user has none until they teach one.
  if (strategies.length === 0) {
    return (
      <div className="p-4 max-w-[900px] mx-auto animate-fade-in">
        <div className="mb-5">
          <h1 className="text-xl font-bold text-[#F1F5F9]">Strategies</h1>
          <p className="text-xs text-[#64748B] mt-0.5">Your personal strategy DNA library</p>
        </div>
        <div className="card p-8 text-center">
          <div className="text-3xl mb-3">📈</div>
          <h2 className="text-sm font-semibold text-[#F1F5F9] mb-1">No strategies yet</h2>
          <p className="text-xs text-[#64748B] max-w-md mx-auto leading-relaxed">
            You haven't taught any strategies yet. Teach one (chart screenshots
            or plain English), or start from a template during onboarding — your
            strategy library appears here.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 max-w-[900px] mx-auto animate-fade-in">
      <div className="mb-5">
        <h1 className="text-xl font-bold text-[#F1F5F9]">Strategies</h1>
        <p className="text-xs text-[#64748B] mt-0.5">Your personal strategy DNA library</p>
      </div>

      <div className="space-y-4">
        {strategies.map((s) => (
          <div key={s.id} className="card p-5">
            <div className="flex items-start justify-between mb-3">
              <div>
                <h3 className="text-base font-bold text-[#F1F5F9]">{s.display_name || s.name}</h3>
                <p className="text-xs text-[#64748B] mt-0.5">{s.description}</p>
                <div className="flex gap-1.5 mt-2">
                  {(s.regimes || []).map((r: string) => (
                    <span key={r} className="text-[10px] px-2 py-0.5 rounded bg-[#3B82F6]/10 text-[#3B82F6] capitalize">
                      {r}
                    </span>
                  ))}
                </div>
              </div>
              <div className="text-right">
                <div
                  className="text-2xl font-bold font-num"
                  style={{ color: (s.metrics?.pnl ?? 0) >= 0 ? "#10B981" : "#EF4444" }}
                >
                  {(s.metrics?.pnl ?? 0) >= 0 ? "+" : ""}₹{Math.abs(s.metrics?.pnl ?? 0).toLocaleString()}
                </div>
                <div className="text-[10px] text-[#64748B]">Total P&L</div>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3 mb-3">
              {[
                { label: "Win Rate", value: `${s.metrics?.win_rate ?? 0}%`, color: (s.metrics?.win_rate ?? 0) >= 55 ? "#10B981" : "#F59E0B" },
                { label: "Avg R", value: `${s.metrics?.avg_r ?? 0}R`, color: "#3B82F6" },
                { label: "Trades", value: String(s.metrics?.trades ?? 0), color: "#94A3B8" },
              ].map((m) => (
                <div key={m.label} className="bg-[#0F1420] rounded-lg p-3 text-center">
                  <div className="text-lg font-bold font-num" style={{ color: m.color }}>{m.value}</div>
                  <div className="text-[10px] text-[#64748B]">{m.label}</div>
                </div>
              ))}
            </div>

            {/* Conditions */}
            <div className="border-t border-[#2A3050] pt-3">
              <div className="text-[10px] uppercase tracking-wide text-[#64748B] mb-2">Conditions</div>
              <div className="flex flex-wrap gap-1.5">
                {(s.conditions || []).map((c: string, i: number) => (
                  <span key={i} className="text-[11px] px-2 py-1 rounded bg-[#8B5CF6]/10 text-[#A78BFA] border border-[#8B5CF6]/20">
                    {c}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
