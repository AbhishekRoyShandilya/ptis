// src/pages/StrategyDetailPage.tsx
import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { strategiesApi } from "@/services/api";

function Badge({ text, colour }: { text: string; colour?: string }) {
  return (
    <span
      className="text-[10px] px-2 py-0.5 rounded-full font-medium"
      style={{ background: colour ? `${colour}22` : "transparent", color: colour || "#94A3B8",
               border: `1px solid ${colour || "#334155"}` }}
    >
      {text}
    </span>
  );
}

function MetricCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-[#0F172A] rounded-xl p-3 flex flex-col gap-0.5">
      <span className="text-[10px] text-[#64748B] uppercase tracking-wider">{label}</span>
      <span className="text-xl font-bold text-[#F1F5F9]">{value}</span>
      {sub && <span className="text-[10px] text-[#64748B]">{sub}</span>}
    </div>
  );
}

const REGIME_COLOUR: Record<string, string> = {
  trending: "#10B981",
  trending_up: "#10B981",
  trending_down: "#EF4444",
  ranging: "#3B82F6",
  volatile: "#F59E0B",
};

export default function StrategyDetailPage() {
  const { name } = useParams<{ name: string }>();
  const navigate = useNavigate();
  const [detail, setDetail] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (name) {
      strategiesApi.get(name)
        .then(setDetail)
        .catch(() => setDetail(null))
        .finally(() => setLoading(false));
    }
  }, [name]);

  if (loading) {
    return (
      <div className="p-4 max-w-[900px] mx-auto">
        <div className="text-[#64748B] text-sm animate-pulse">Loading strategy…</div>
      </div>
    );
  }

  // No demo detail — show an honest empty state if the strategy has no data.
  if (!detail || detail.error) {
    return (
      <div className="p-4 max-w-[900px] mx-auto animate-fade-in">
        <button onClick={() => navigate(-1)}
          className="text-xs text-[#64748B] hover:text-[#94A3B8] mb-4">← Back</button>
        <div className="card p-8 text-center">
          <div className="text-3xl mb-3">📈</div>
          <h2 className="text-sm font-semibold text-[#F1F5F9] mb-1">No data for this strategy yet</h2>
          <p className="text-xs text-[#64748B] max-w-md mx-auto leading-relaxed">
            This strategy's performance and rules appear once it has been taught
            and has trades. Teach or trade this strategy to populate it.
          </p>
        </div>
      </div>
    );
  }

  const d = detail;
  const m = d.metrics || {};

  return (
    <div className="p-4 max-w-[900px] mx-auto animate-fade-in">
      {/* Back button */}
      <button
        onClick={() => navigate(-1)}
        className="flex items-center gap-1 text-xs text-[#64748B] hover:text-[#94A3B8] mb-4"
      >
        ← Back to Strategies
      </button>

      {/* Header */}
      <div className="mb-5 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <div>
          <h1 className="text-xl font-bold text-[#F1F5F9]">{d.display_name || d.name}</h1>
          {d.description && (
            <p className="text-xs text-[#64748B] mt-0.5">{d.description}</p>
          )}
        </div>
        <div className="flex flex-wrap gap-1.5">
          {(d.regimes || []).map((r: string) => (
            <Badge key={r} text={r.replace(/_/g, " ")} colour={REGIME_COLOUR[r]} />
          ))}
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <MetricCard label="Win Rate" value={`${((m.win_rate || 0) * 100).toFixed(0)}%`}
                    sub={`${m.trades || 0} trades`} />
        <MetricCard label="Avg R" value={(m.avg_r || 0).toFixed(2)}
                    sub="per closed trade" />
        <MetricCard label="Profit Factor" value={(m.profit_factor || 0).toFixed(2)} />
        <MetricCard label="Total P&L" value={`₹${((m.total_pnl || 0) / 1000).toFixed(1)}k`} />
      </div>

      {/* Edge Health */}
      <div className={`rounded-xl p-3 mb-4 flex items-start gap-3 ${
        d.edge_health?.stable ? "bg-[#10B98111] border border-[#10B98133]" : "bg-[#EF444411] border border-[#EF444433]"
      }`}>
        <span className="text-lg">{d.edge_health?.stable ? "✅" : "⚠️"}</span>
        <div>
          <p className="text-xs font-semibold text-[#F1F5F9]">
            Edge {d.edge_health?.stable ? "Stable" : "Under Pressure"}
          </p>
          <p className="text-[10px] text-[#94A3B8] mt-0.5">
            {d.edge_health?.decay_signal ||
              `90d win rate: ${((d.edge_health?.win_rate_90d || 0) * 100).toFixed(0)}%  →  30d win rate: ${((d.edge_health?.win_rate_30d || 0) * 100).toFixed(0)}%`}
          </p>
        </div>
      </div>

      {/* Conditions & Exits */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
        {d.conditions && (
          <div className="bg-[#1E293B] rounded-xl p-4">
            <h2 className="text-sm font-semibold text-[#F1F5F9] mb-3">Entry Conditions</h2>
            <ul className="space-y-2">
              {d.conditions.map((c: string, i: number) => (
                <li key={i} className="flex gap-2 text-xs text-[#94A3B8]">
                  <span className="text-[#3B82F6] mt-0.5 shrink-0">{i + 1}.</span>
                  {c}
                </li>
              ))}
            </ul>
          </div>
        )}
        {d.exits && (
          <div className="bg-[#1E293B] rounded-xl p-4">
            <h2 className="text-sm font-semibold text-[#F1F5F9] mb-3">Exit Rules</h2>
            <ul className="space-y-2">
              {d.exits.map((e: string, i: number) => (
                <li key={i} className="flex gap-2 text-xs text-[#94A3B8]">
                  <span className="text-[#10B981] mt-0.5 shrink-0">→</span>
                  {e}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Recent Trades */}
      {d.recent_trades && d.recent_trades.length > 0 && (
        <div className="bg-[#1E293B] rounded-xl p-4">
          <h2 className="text-sm font-semibold text-[#F1F5F9] mb-3">Recent Trades</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-[#64748B] border-b border-[#334155]">
                  <th className="text-left py-1.5 font-medium">Date</th>
                  <th className="text-left py-1.5 font-medium">Symbol</th>
                  <th className="text-right py-1.5 font-medium">P&L</th>
                  <th className="text-right py-1.5 font-medium">R</th>
                  <th className="text-right py-1.5 font-medium">Result</th>
                </tr>
              </thead>
              <tbody>
                {d.recent_trades.map((t: any, i: number) => (
                  <tr key={i} className="border-b border-[#1E293B] hover:bg-[#0F172A]">
                    <td className="py-1.5 text-[#94A3B8]">{t.date}</td>
                    <td className="py-1.5 text-[#F1F5F9] font-medium">{t.symbol}</td>
                    <td className={`py-1.5 text-right font-mono ${t.pnl >= 0 ? "text-[#10B981]" : "text-[#EF4444]"}`}>
                      {t.pnl >= 0 ? "+" : ""}₹{t.pnl.toLocaleString()}
                    </td>
                    <td className={`py-1.5 text-right font-mono ${t.r >= 0 ? "text-[#10B981]" : "text-[#EF4444]"}`}>
                      {t.r >= 0 ? "+" : ""}{t.r.toFixed(1)}R
                    </td>
                    <td className="py-1.5 text-right">
                      <Badge
                        text={t.result}
                        colour={t.result === "win" ? "#10B981" : "#EF4444"}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {loading && (
        <div className="fixed inset-0 flex items-center justify-center bg-black/30 z-50">
          <div className="text-[#64748B] text-sm animate-pulse">Loading strategy…</div>
        </div>
      )}
    </div>
  );
}
