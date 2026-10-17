// src/pages/BehaviorPage.tsx
import { useEffect, useState } from "react";
import { behaviorApi } from "@/services/api";

const BIAS_COLORS: Record<string, string> = {
  overtrading: "#EF4444",
  revenge_trading: "#EF4444",
  fomo: "#F59E0B",
  early_exit: "#F59E0B",
  loss_aversion: "#8B5CF6",
  confirmation_bias: "#3B82F6",
};

export default function BehaviorPage() {
  const [baseline, setBaseline] = useState<any>(null);
  const [improvement, setImprovement] = useState<any>(null);
  const [biases, setBiases] = useState<any[]>([]);
  const [deviations, setDeviations] = useState<any[]>([]);
  const [discipline, setDiscipline] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      behaviorApi.baseline().then(setBaseline).catch(() => {}),
      behaviorApi.improvement().then(setImprovement).catch(() => {}),
      behaviorApi.biases().then((d: any) => setBiases(Array.isArray(d) ? d : [])).catch(() => {}),
      behaviorApi.deviations().then((d: any) => setDeviations(Array.isArray(d) ? d : [])).catch(() => {}),
      behaviorApi.disciplineScore().then(setDiscipline).catch(() => {}),
    ]).finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="p-4 max-w-[1200px] mx-auto">
        <div className="text-[#64748B] text-sm animate-pulse">Loading behavioral data…</div>
      </div>
    );
  }

  // No demo data — behavioral intelligence is built only from real trades.
  const hasData = biases.length > 0 || deviations.length > 0;
  if (!hasData) {
    return (
      <div className="p-4 max-w-[1200px] mx-auto animate-fade-in">
        <div className="mb-5">
          <h1 className="text-xl font-bold text-[#F1F5F9]">Behavioral Intelligence</h1>
          <p className="text-xs text-[#64748B] mt-0.5">Pattern recognition and psychological profiling</p>
        </div>
        <div className="card p-8 text-center">
          <div className="text-3xl mb-3">🧠</div>
          <h2 className="text-sm font-semibold text-[#F1F5F9] mb-1">No behavioral data yet</h2>
          <p className="text-xs text-[#64748B] max-w-md mx-auto leading-relaxed">
            Your biases, deviations, and discipline patterns are detected from
            your real trades. Sync your broker history or log trades, and your
            behavioral profile builds automatically.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 max-w-[1200px] mx-auto animate-fade-in">
      <div className="mb-5">
        <h1 className="text-xl font-bold text-[#F1F5F9]">Behavioral Intelligence</h1>
        <p className="text-xs text-[#64748B] mt-0.5">Pattern recognition and psychological profiling</p>
      </div>

      <div className="grid grid-cols-12 gap-4">
        {/* Bias Detector */}
        <div className="col-span-12 lg:col-span-7 card p-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94A3B8] mb-4">
            Detected Biases
          </h3>
          <div className="space-y-4">
            {biases.map((b, i) => {
              const color = Object.entries(BIAS_COLORS).find(([k]) =>
                b.bias?.toLowerCase().includes(k) ||
                b.description?.toLowerCase().includes(k)
              )?.[1] ?? "#64748B";

              return (
                <div key={i} className="border-b border-[#2A3050] pb-4 last:border-0 last:pb-0">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <div className="text-sm font-semibold text-[#F1F5F9]">{b.bias || b.name}</div>
                      <div className="text-[11px] text-[#64748B] mt-0.5">{b.description}</div>
                    </div>
                    <span
                      className="text-xs font-bold font-num ml-3 shrink-0"
                      style={{ color: "#EF4444" }}
                    >
                      -₹{(b.cost_estimate ?? 0).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-[#64748B] w-24 shrink-0">Frequency: {b.frequency}%</span>
                    <div className="flex-1 h-1.5 bg-[#0A0E17] rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{ width: `${b.frequency ?? 0}%`, backgroundColor: color, opacity: 0.8 }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Improvement tracker */}
        <div className="col-span-12 lg:col-span-5 space-y-4">
          {/* Discipline Score Detail */}
          <div className="card p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94A3B8] mb-3">
              Discipline Score
            </h3>
            <div className="flex items-baseline gap-1 mb-2">
              <span className="text-3xl font-bold font-num text-[#3B82F6]">
                {discipline?.score != null ? Number(discipline.score).toFixed(1) : "—"}
              </span>
              <span className="text-sm text-[#64748B]">/10</span>
              {discipline?.trend && discipline.trend !== "insufficient_data" && (
                <span
                  className="text-xs ml-auto"
                  style={{ color: discipline.trend === "improving" ? "#10B981" : discipline.trend === "declining" ? "#EF4444" : "#64748B" }}
                >
                  {discipline.trend === "improving" ? "↑ improving" : discipline.trend === "declining" ? "↓ declining" : "→ stable"}
                </span>
              )}
            </div>
            <div className="h-2 bg-[#0A0E17] rounded-full overflow-hidden mb-3">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{ width: `${((discipline?.score ?? 0) / 10) * 100}%`, background: "linear-gradient(90deg, #8B5CF6, #3B82F6)" }}
              />
            </div>
            <div className="space-y-1.5 text-[10px] text-[#94A3B8]">
              <div className="flex items-center justify-between">
                <span>Trades scored</span>
                <span className="font-num">{discipline?.trades_scored ?? 0}</span>
              </div>
              {discipline?.high != null && (
                <div className="flex items-center justify-between">
                  <span>Best</span>
                  <span className="font-num text-[#10B981]">{Number(discipline.high).toFixed(1)}</span>
                </div>
              )}
              {discipline?.low != null && (
                <div className="flex items-center justify-between">
                  <span>Lowest</span>
                  <span className="font-num text-[#EF4444]">{Number(discipline.low).toFixed(1)}</span>
                </div>
              )}
            </div>
          </div>

          {/* Focus area — derived from your costliest detected bias */}
          {biases.length > 0 && (() => {
            const top = [...biases].sort((a, b) => (b.cost_estimate ?? 0) - (a.cost_estimate ?? 0))[0];
            return (
              <div className="card p-4" style={{ borderColor: "#8B5CF6" + "30", background: "rgba(139,92,246,0.05)" }}>
                <div className="flex items-center gap-1.5 mb-2">
                  <span className="text-sm">🎯</span>
                  <span className="text-xs font-semibold text-[#A78BFA]">This Week's Focus</span>
                </div>
                <p className="text-xs text-[#CBD5E1] leading-relaxed">
                  Reduce <span className="text-[#F1F5F9] font-semibold">{top.bias || top.name}</span>
                  {top.cost_estimate ? <> — it's cost you <span className="text-[#EF4444] font-semibold">₹{Number(top.cost_estimate).toLocaleString()}</span></> : null}.
                  {top.description ? <span className="text-[#94A3B8]"> {top.description}</span> : null}
                </p>
              </div>
            );
          })()}
        </div>

        {/* Deviation Timeline */}
        <div className="col-span-12 card p-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94A3B8] mb-3">
            Recent Deviations
          </h3>
          <div className="space-y-2">
            {deviations.map((d) => {
              const color =
                d.severity === "high" ? "#EF4444" :
                d.severity === "medium" ? "#F59E0B" : "#3B82F6";
              return (
                <div key={d.id} className="flex items-start gap-3 py-2.5 border-b border-[#2A3050] last:border-0">
                  <div
                    className="h-2 w-2 rounded-full mt-1.5 shrink-0"
                    style={{ backgroundColor: color }}
                  />
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-[#F1F5F9]">{d.symbol}</span>
                      <span
                        className="text-[9px] px-1.5 py-0.5 rounded"
                        style={{ color, background: color + "15" }}
                      >
                        {d.deviation_type?.replace("_", " ")}
                      </span>
                    </div>
                    <div className="text-[11px] text-[#94A3B8] mt-0.5">{d.description}</div>
                    <div className="text-[10px] text-[#3A4060] mt-0.5">
                      {new Date(d.timestamp).toLocaleString("en-IN", {
                        dateStyle: "short",
                        timeStyle: "short",
                      })}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
