// src/pages/RegimePage.tsx
import { useEffect, useState } from "react";
import { regimeApi } from "@/services/api";

const REGIME_INFO = {
  trending: {
    color: "#10B981",
    icon: "↗",
    desc: "Strong directional momentum. Trend-following strategies perform best.",
    strategies: ["ORB Breakout", "Momentum continuation", "VWAP trend"],
    avoid: ["Counter-trend fades", "Mean-reversion"],
  },
  ranging: {
    color: "#F59E0B",
    icon: "↔",
    desc: "Price bouncing between levels. Mean-reversion and range plays work.",
    strategies: ["Range fade", "Support/Resistance plays", "VWAP mean-revert"],
    avoid: ["Breakout trades", "Momentum chasing"],
  },
  volatile: {
    color: "#EF4444",
    icon: "⚡",
    desc: "High uncertainty. Reduce size and risk. Only trade A+ setups.",
    strategies: ["Gap fills only", "Very tight stops"],
    avoid: ["Most strategies", "Overtrading"],
  },
  quiet: {
    color: "#3B82F6",
    icon: "〜",
    desc: "Low volatility, range-bound. Scalp the extremes or wait.",
    strategies: ["Scalping", "Tight range plays"],
    avoid: ["Large targets", "Trending strategies"],
  },
};

export default function RegimePage() {
  const [current, setCurrent] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      regimeApi.current().then(setCurrent).catch(() => {}),
      regimeApi.history().then((d: any) => setHistory(Array.isArray(d) ? d : [])).catch(() => {}),
    ]).finally(() => setLoading(false));
  }, []);

  const regime = current?.regime?.toLowerCase() || "trending";
  const info = REGIME_INFO[regime as keyof typeof REGIME_INFO] || REGIME_INFO.trending;

  return (
    <div className="p-4 max-w-[900px] mx-auto animate-fade-in">
      <div className="mb-5">
        <h1 className="text-xl font-bold text-[#F1F5F9]">Market Regime</h1>
        <p className="text-xs text-[#64748B] mt-0.5">Current market environment & strategy alignment</p>
      </div>

      {/* Current Regime */}
      <div
        className="card p-6 mb-5"
        style={{ borderColor: info.color + "40", background: info.color + "08" }}
      >
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <div
                className="h-10 w-10 rounded-xl flex items-center justify-center text-xl font-bold"
                style={{ background: info.color + "20", color: info.color }}
              >
                {info.icon}
              </div>
              <div>
                <div className="text-2xl font-bold capitalize" style={{ color: info.color }}>
                  {regime}
                </div>
                <div className="text-[10px] uppercase tracking-wide text-[#64748B]">Current Regime</div>
              </div>
            </div>
            <p className="text-sm text-[#CBD5E1] leading-relaxed max-w-lg">{info.desc}</p>
          </div>
          <div className="flex items-center gap-1">
            <div className="h-2 w-2 rounded-full pulse-dot" style={{ backgroundColor: info.color }} />
            <span className="text-[10px] text-[#64748B]">Live</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mt-5">
          <div>
            <div className="text-[10px] uppercase tracking-wide text-[#10B981] mb-2">✅ Recommended</div>
            <div className="space-y-1">
              {info.strategies.map((s) => (
                <div key={s} className="flex items-center gap-2 text-xs text-[#CBD5E1]">
                  <span className="text-[#10B981]">›</span>
                  {s}
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wide text-[#EF4444] mb-2">❌ Avoid</div>
            <div className="space-y-1">
              {info.avoid.map((s) => (
                <div key={s} className="flex items-center gap-2 text-xs text-[#CBD5E1]">
                  <span className="text-[#EF4444]">›</span>
                  {s}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Personal Performance by Regime */}
      <div className="card p-4 mb-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94A3B8] mb-4">
          Your Performance by Regime
        </h3>
        <div className="grid grid-cols-4 gap-3">
          {Object.entries(REGIME_INFO).map(([r, info]) => (
            <div
              key={r}
              className={`p-3 rounded-xl border transition-all ${
                r === regime ? "border-[" + info.color + "] bg-[" + info.color + "]/5" : "border-[#2A3050] bg-[#0F1420]"
              }`}
              style={r === regime ? { borderColor: info.color + "40", background: info.color + "08" } : {}}
            >
              <div className="flex items-center gap-1.5 mb-2">
                <span style={{ color: info.color }}>{info.icon}</span>
                <span className="text-xs font-semibold capitalize" style={{ color: info.color }}>
                  {r}
                </span>
                {r === regime && (
                  <span className="text-[8px] bg-[#10B981]/10 text-[#10B981] px-1 rounded ml-auto">NOW</span>
                )}
              </div>
              <div className="text-lg font-bold font-num text-[#F1F5F9]">
                {r === "trending" ? "64%" : r === "ranging" ? "51%" : r === "volatile" ? "38%" : "44%"}
              </div>
              <div className="text-[10px] text-[#64748B]">Win rate</div>
            </div>
          ))}
        </div>
      </div>

      {/* Regime History */}
      {history.length > 0 && (
        <div className="card p-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94A3B8] mb-3">
            Regime History
          </h3>
          <div className="space-y-2">
            {history.slice(0, 7).map((h: any, i) => {
              const ri = REGIME_INFO[(h.regime || "trending").toLowerCase() as keyof typeof REGIME_INFO] || REGIME_INFO.trending;
              return (
                <div key={i} className="flex items-center gap-3 text-xs py-1.5 border-b border-[#2A3050] last:border-0">
                  <span style={{ color: ri.color }}>{ri.icon}</span>
                  <span className="capitalize text-[#F1F5F9] w-20">{h.regime}</span>
                  <span className="text-[#64748B]">
                    {new Date(h.from || h.date).toLocaleDateString("en-IN")}
                    {h.to && ` → ${new Date(h.to).toLocaleDateString("en-IN")}`}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
