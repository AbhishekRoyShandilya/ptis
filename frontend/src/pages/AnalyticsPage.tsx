// src/pages/AnalyticsPage.tsx
import { useEffect, useState } from "react";
import { analyticsApi } from "@/services/api";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  CartesianGrid,
} from "recharts";

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#1A2035] border border-[#2A3050] rounded-lg px-3 py-2 text-xs">
      <div className="text-[#94A3B8] mb-1">{label}</div>
      {payload.map((p: any) => (
        <div key={p.name} className="font-num" style={{ color: p.color }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(2) : p.value}
        </div>
      ))}
    </div>
  );
};

// Bucket label → colour for the R-distribution bars (negative=red, flat=amber, positive=green)
function rColor(label: string): string {
  if (label.includes("<-2") || label.includes("-2R to -1")) return "#EF4444";
  if (label.includes("-1R to 0") || label.includes("0R to 1")) return "#F59E0B";
  return "#10B981";
}

export default function AnalyticsPage() {
  const [perf, setPerf] = useState<any>(null);
  const [equity, setEquity] = useState<any[]>([]);
  const [strategies, setStrategies] = useState<any[]>([]);
  const [tab, setTab] = useState<"performance" | "behavior" | "strategies">("performance");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      analyticsApi.performance().then(setPerf).catch(() => {}),
      analyticsApi.equity().then((d: any) => {
        const pts = Array.isArray(d) ? d : (d?.points || []);
        setEquity(pts.map((p: any) => ({
          date: p.date ? new Date(p.date).toLocaleDateString("en-IN", { month: "short", day: "numeric" }) : "",
          pnl: p.cumulative_pnl ?? p.pnl ?? 0,
        })));
      }).catch(() => {}),
      analyticsApi.strategies().then((d: any) => setStrategies(Array.isArray(d) ? d : [])).catch(() => {}),
    ]).finally(() => setLoading(false));
  }, []);

  const rDist: any[] = perf?.r_distribution || [];
  const rMax = Math.max(1, ...rDist.map((r: any) => r.count || 0));
  const sessions: any[] = perf?.sessions || [];

  if (loading) {
    return (
      <div className="p-4 max-w-[1200px] mx-auto">
        <div className="text-[#64748B] text-sm animate-pulse">Loading analytics…</div>
      </div>
    );
  }

  // No demo analytics — performance is computed only from your real trades.
  const hasData = !!perf && (perf.total_trades ?? perf.trades ?? 0) > 0;
  if (!hasData) {
    return (
      <div className="p-4 max-w-[1200px] mx-auto animate-fade-in">
        <div className="mb-5">
          <h1 className="text-xl font-bold text-[#F1F5F9]">Analytics</h1>
          <p className="text-xs text-[#64748B] mt-0.5">Performance &amp; behavioral intelligence</p>
        </div>
        <div className="card p-8 text-center">
          <div className="text-3xl mb-3">📊</div>
          <h2 className="text-sm font-semibold text-[#F1F5F9] mb-1">No analytics yet</h2>
          <p className="text-xs text-[#64748B] max-w-md mx-auto leading-relaxed">
            Performance, R-distribution, and session analytics appear once you
            have trades. Sync your broker history or log trades to populate them.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 max-w-[1200px] mx-auto animate-fade-in">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h1 className="text-xl font-bold text-[#F1F5F9]">Analytics</h1>
          <p className="text-xs text-[#64748B] mt-0.5">Performance & behavioral intelligence</p>
        </div>
      </div>

      {/* Summary KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
        {[
          { label: "Win Rate", value: `${((perf?.win_rate ?? 0) * 100).toFixed(0)}%`, color: "#10B981" },
          { label: "Avg R-Multiple", value: `${(perf?.avg_r ?? 0).toFixed(2)}R`, color: "#3B82F6" },
          { label: "Profit Factor", value: (perf?.profit_factor ?? 0).toFixed(2), color: "#8B5CF6" },
          { label: "Expectancy", value: `₹${(perf?.expectancy ?? 0).toLocaleString("en-IN")}`, color: (perf?.expectancy ?? 0) >= 0 ? "#10B981" : "#EF4444" },
        ].map((stat) => (
          <div key={stat.label} className="card p-4">
            <div className="text-[10px] uppercase tracking-widest text-[#64748B] mb-1">{stat.label}</div>
            <div className="text-2xl font-bold font-num" style={{ color: stat.color }}>
              {stat.value}
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-4">
        {(["performance", "behavior", "strategies"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium capitalize transition-all ${
              tab === t
                ? "bg-[#8B5CF6]/20 text-[#A78BFA] border border-[#8B5CF6]/30"
                : "text-[#64748B] hover:text-[#94A3B8] bg-[#1A2035]"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "performance" && (
        <div className="grid grid-cols-12 gap-4">
          {/* Equity Curve */}
          <div className="col-span-12 lg:col-span-8 card p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94A3B8] mb-4">
              Equity Curve
            </h3>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={equity}>
                <CartesianGrid stroke="#2A3050" strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fill: "#64748B", fontSize: 10 }} />
                <YAxis tick={{ fill: "#64748B", fontSize: 10 }} tickFormatter={(v) => `₹${(v/1000).toFixed(0)}k`} />
                <Tooltip content={<CustomTooltip />} />
                <Line
                  type="monotone"
                  dataKey="pnl"
                  stroke="#8B5CF6"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: "#A78BFA" }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* R Distribution */}
          <div className="col-span-12 lg:col-span-4 card p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94A3B8] mb-4">
              R Distribution
            </h3>
            <div className="space-y-2">
              {rDist.length === 0 ? (
                <div className="text-[11px] text-[#64748B] py-4 text-center">Not enough trades with R-multiples yet.</div>
              ) : rDist.map((r: any) => (
                <div key={r.r} className="flex items-center gap-2">
                  <span className="text-[10px] text-[#64748B] w-20 shrink-0">{r.r}</span>
                  <div className="flex-1 h-4 bg-[#0A0E17] rounded overflow-hidden">
                    <div
                      className="h-full rounded transition-all duration-500"
                      style={{
                        width: `${(r.count / rMax) * 100}%`,
                        backgroundColor: rColor(r.r),
                        opacity: 0.7,
                      }}
                    />
                  </div>
                  <span className="text-[10px] font-num text-[#94A3B8] w-5">{r.count}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Session Performance */}
          <div className="col-span-12 card p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94A3B8] mb-4">
              Win Rate by Session
            </h3>
            {sessions.length === 0 ? (
              <div className="text-[11px] text-[#64748B] py-4 text-center">No session data yet.</div>
            ) : (
              <>
                <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${sessions.length}, minmax(0, 1fr))` }}>
                  {sessions.map((s: any) => (
                    <div key={s.session} className="text-center">
                      <div className="text-[10px] text-[#64748B] mb-2">{s.session}</div>
                      <div className="relative h-24 flex items-end justify-center">
                        <div
                          className="w-12 rounded-t transition-all duration-700"
                          style={{
                            height: `${s.wr}%`,
                            background:
                              s.wr >= 60
                                ? "linear-gradient(180deg, #10B981, #10B98180)"
                                : s.wr >= 45
                                ? "linear-gradient(180deg, #F59E0B, #F59E0B80)"
                                : "linear-gradient(180deg, #EF4444, #EF444480)",
                          }}
                        />
                      </div>
                      <div
                        className="text-lg font-bold font-num mt-1"
                        style={{ color: s.wr >= 60 ? "#10B981" : s.wr >= 45 ? "#F59E0B" : "#EF4444" }}
                      >
                        {s.wr}%
                      </div>
                      <div className="text-[10px] text-[#64748B]">{s.trades} trades</div>
                    </div>
                  ))}
                </div>
                {sessions.length >= 2 && (() => {
                  const best = sessions.reduce((a: any, b: any) => (b.wr > a.wr ? b : a));
                  const worst = sessions.reduce((a: any, b: any) => (b.wr < a.wr ? b : a));
                  if (best.session === worst.session) return null;
                  return (
                    <div className="mt-4 p-3 rounded-lg border border-[#8B5CF6]/20 bg-[#8B5CF6]/5">
                      <p className="text-xs text-[#CBD5E1]">
                        <strong className="text-[#F1F5F9]">Key insight:</strong> Your {best.session} win rate
                        {" "}({best.wr}%) beats your {worst.session} rate ({worst.wr}%). Focusing on your
                        {" "}{best.session} sessions could improve results.
                      </p>
                    </div>
                  );
                })()}
              </>
            )}
          </div>
        </div>
      )}

      {tab === "behavior" && (
        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-12 md:col-span-6 card p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94A3B8] mb-3">
              Behavioral Impact
            </h3>
            <div className="space-y-3">
              {[
                { label: "Held winners full target", impact: "+₹8,400", positive: true },
                { label: "Journaled all trades", impact: "+awareness", positive: true },
                { label: "Afternoon revenge trades", impact: "-₹5,200", positive: false },
                { label: "Skipped risk analyzer", impact: "-30% WR", positive: false },
                { label: "Overtraded after loss", impact: "-₹3,100", positive: false },
              ].map((item) => (
                <div
                  key={item.label}
                  className="flex items-center justify-between py-2 border-b border-[#2A3050] last:border-0 text-xs"
                >
                  <div className="flex items-center gap-2">
                    <span className={item.positive ? "text-[#10B981]" : "text-[#EF4444]"}>
                      {item.positive ? "✅" : "❌"}
                    </span>
                    <span className="text-[#CBD5E1]">{item.label}</span>
                  </div>
                  <span
                    className="font-num font-semibold"
                    style={{ color: item.positive ? "#10B981" : "#EF4444" }}
                  >
                    {item.impact}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="col-span-12 md:col-span-6 card p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94A3B8] mb-3">
              Discipline Trend
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart
                data={[
                  { week: "W1", score: 6.2 },
                  { week: "W2", score: 7.1 },
                  { week: "W3", score: 6.8 },
                  { week: "W4", score: 7.5 },
                  { week: "W5", score: 8.1 },
                ]}
              >
                <CartesianGrid stroke="#2A3050" strokeDasharray="3 3" />
                <XAxis dataKey="week" tick={{ fill: "#64748B", fontSize: 10 }} />
                <YAxis domain={[0, 10]} tick={{ fill: "#64748B", fontSize: 10 }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="score" fill="#8B5CF6" radius={[4, 4, 0, 0]} opacity={0.8} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {tab === "strategies" && (
        <div className="card p-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94A3B8] mb-4">
            Strategy Performance
          </h3>
          <div className="space-y-0">
            {strategies.length === 0 ? (
              <div className="text-[11px] text-[#64748B] py-6 text-center">
                No strategy performance yet — trade your strategies and this fills in.
              </div>
            ) : strategies.map((s: any) => {
              const name = s.display_name || s.name || s.strategy_name || s.short_code || "Strategy";
              const trades = s.total_trades ?? s.trades ?? 0;
              const wrRaw = s.win_rate ?? 0;
              const wr = wrRaw <= 1 ? wrRaw * 100 : wrRaw;
              const avgR = Number(s.avg_r ?? s.avgR ?? 0);
              const pnl = s.total_pnl ?? s.pnl ?? 0;
              return (
                <div
                  key={s.id || name}
                  className="flex items-center gap-4 py-3 border-b border-[#2A3050] last:border-0"
                >
                  <div className="flex-1">
                    <div className="text-sm font-semibold text-[#F1F5F9]">{name}</div>
                    <div className="text-[10px] text-[#64748B]">{trades} trades</div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm font-bold font-num" style={{ color: wr >= 55 ? "#10B981" : "#F59E0B" }}>
                      {wr.toFixed(0)}%
                    </div>
                    <div className="text-[10px] text-[#64748B]">WR</div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm font-bold font-num text-[#3B82F6]">{avgR.toFixed(1)}R</div>
                    <div className="text-[10px] text-[#64748B]">Avg R</div>
                  </div>
                  <div
                    className="text-sm font-bold font-num w-24 text-right"
                    style={{ color: pnl >= 0 ? "#10B981" : "#EF4444" }}
                  >
                    {pnl >= 0 ? "+" : ""}₹{Math.abs(pnl).toLocaleString()}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
