// src/pages/ShadowTradingPage.tsx
import { useEffect, useState } from "react";
import { shadowApi, type ShadowTrade, type ShadowSession } from "@/services/api";
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, Legend, CartesianGrid } from "recharts";

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const diff = (payload[1]?.value ?? 0) - (payload[0]?.value ?? 0);
  return (
    <div className="bg-[#1A2035] border border-[#2A3050] rounded-lg px-3 py-2 text-xs">
      <div className="text-[#94A3B8] mb-1.5">{label}</div>
      {payload.map((p: any) => (
        <div key={p.name} className="font-num flex justify-between gap-4" style={{ color: p.color }}>
          <span>{p.name}:</span>
          <span>₹{(p.value as number).toLocaleString()}</span>
        </div>
      ))}
      <div className="mt-1.5 pt-1.5 border-t border-[#2A3050] font-num text-[#F59E0B] flex justify-between gap-4">
        <span>Gap:</span>
        <span>₹{diff.toLocaleString()}</span>
      </div>
    </div>
  );
};

const Empty = ({ text }: { text: string }) => (
  <div className="text-center py-8 text-xs text-[#64748B]">{text}</div>
);

export default function ShadowTradingPage() {
  const [sessions, setSessions] = useState<ShadowSession[]>([]);
  const [trades, setTrades] = useState<ShadowTrade[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      shadowApi.sessions().then(setSessions).catch(() => {}),
      shadowApi.trades().then(setTrades).catch(() => {}),
      shadowApi.summary().then(setSummary).catch(() => {}),
    ]).finally(() => setLoading(false));
  }, []);

  // Real data only — no demo fallbacks. Absent fields render as 0 / empty-state.
  const realTotal = Number(summary?.real_total ?? 0);
  const shadowTotal = Number(summary?.shadow_total ?? 0);
  const gap = shadowTotal - realTotal;
  const chartData: any[] = Array.isArray(summary?.comparison) ? summary.comparison : [];
  const quality: { label: string; score: number }[] = Array.isArray(summary?.quality) ? summary.quality : [];
  const hasData = trades.length > 0 || sessions.length > 0 || summary != null;

  return (
    <div className="p-4 max-w-[1200px] mx-auto animate-fade-in">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h1 className="text-xl font-bold text-[#F1F5F9]">Shadow Trading</h1>
          <p className="text-xs text-[#64748B] mt-0.5">Real vs. optimal execution comparison</p>
        </div>
      </div>

      {loading ? (
        <Empty text="Loading shadow data…" />
      ) : !hasData ? (
        <div className="card p-8 text-center">
          <div className="text-3xl text-[#3A4060] mb-3">◱</div>
          <p className="text-sm text-[#94A3B8]">No shadow trades recorded yet.</p>
          <p className="text-xs text-[#64748B] mt-1">
            Shadow comparisons appear here once real trades are logged and shadowed.
          </p>
        </div>
      ) : (
        <>
          {/* Summary */}
          <div className="grid grid-cols-3 gap-3 mb-5">
            <div className="card p-4">
              <div className="text-[10px] uppercase tracking-widest text-[#64748B] mb-1">Real P&L</div>
              <div className="text-2xl font-bold font-num text-[#3B82F6]">₹{realTotal.toLocaleString()}</div>
              <div className="text-xs text-[#64748B] mt-0.5">Your actual performance</div>
            </div>
            <div className="card p-4">
              <div className="text-[10px] uppercase tracking-widest text-[#64748B] mb-1">Shadow P&L</div>
              <div className="text-2xl font-bold font-num text-[#8B5CF6]">₹{shadowTotal.toLocaleString()}</div>
              <div className="text-xs text-[#64748B] mt-0.5">What AI/strategy would achieve</div>
            </div>
            <div className="card p-4 border-[#F59E0B]/30 glow-amber">
              <div className="text-[10px] uppercase tracking-widest text-[#64748B] mb-1">Execution Gap</div>
              <div className="text-2xl font-bold font-num text-[#F59E0B]">₹{gap.toLocaleString()}</div>
              <div className="text-xs text-[#64748B] mt-0.5">Behavioral cost</div>
            </div>
          </div>

          {/* Cumulative Comparison Chart */}
          <div className="card p-4 mb-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94A3B8] mb-4">
              Cumulative P&L Comparison
            </h3>
            {chartData.length === 0 ? (
              <Empty text="No comparison series yet." />
            ) : (
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chartData}>
                  <CartesianGrid stroke="#2A3050" strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fill: "#64748B", fontSize: 10 }} />
                  <YAxis tick={{ fill: "#64748B", fontSize: 10 }} tickFormatter={(v) => `₹${(v / 1000).toFixed(0)}k`} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend
                    formatter={(v) => (
                      <span className="text-[10px] text-[#94A3B8]">{v === "real" ? "Your P&L" : "Shadow P&L"}</span>
                    )}
                  />
                  <Line type="monotone" dataKey="real" stroke="#3B82F6" strokeWidth={2} dot={false} name="real" />
                  <Line type="monotone" dataKey="shadow" stroke="#8B5CF6" strokeWidth={2} dot={false} strokeDasharray="5 3" name="shadow" />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Divergence analysis */}
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-12 md:col-span-7 card p-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94A3B8] mb-3">Shadow Trades</h3>
              {trades.length === 0 ? (
                <Empty text="No shadow trades yet." />
              ) : (
                <div className="space-y-2">
                  {trades.slice(0, 12).map((t) => (
                    <div key={t.id} className="flex items-center gap-3 py-2.5 border-b border-[#2A3050] last:border-0">
                      <span className="text-sm font-bold text-[#F1F5F9] w-20">{t.symbol}</span>
                      <div className="flex-1 text-[11px] text-[#64748B]">{t.type} · {t.status}</div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <div className="text-[10px] text-[#64748B]">P&L</div>
                          <div className="text-xs font-num font-semibold"
                               style={{ color: (t.pnl ?? 0) >= 0 ? "#10B981" : "#EF4444" }}>
                            {(t.pnl ?? 0) >= 0 ? "+" : ""}₹{(t.pnl ?? 0).toLocaleString()}
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-[10px] text-[#64748B]">R</div>
                          <div className="text-xs font-num font-semibold text-[#8B5CF6]">
                            {t.r_multiple != null ? `${t.r_multiple.toFixed(2)}R` : "—"}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="col-span-12 md:col-span-5 card p-4">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94A3B8] mb-3">Execution Quality</h3>
              {quality.length === 0 ? (
                <Empty text="No execution-quality scores yet." />
              ) : (
                <div className="space-y-4">
                  {quality.map((item) => (
                    <div key={item.label}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-[#94A3B8]">{item.label}</span>
                        <span className="font-num font-semibold"
                              style={{ color: item.score >= 75 ? "#10B981" : item.score >= 55 ? "#F59E0B" : "#EF4444" }}>
                          {item.score}%
                        </span>
                      </div>
                      <div className="h-1.5 bg-[#0A0E17] rounded-full overflow-hidden">
                        <div className="h-full rounded-full transition-all duration-700"
                             style={{ width: `${item.score}%`,
                                      background: item.score >= 75 ? "#10B981" : item.score >= 55 ? "#F59E0B" : "#EF4444" }} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
