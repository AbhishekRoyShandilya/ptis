// src/pages/PatternDiscoveryPage.tsx
// Pattern Discovery Console — natural-language → backtested strategy pipeline
//
// Two modes:
//   1. One-Shot: classic flow — describe a pattern, get one backtested strategy
//   2. Continuous: background loop tests many candidates, validated drafts queue
//      up for trader review and "Send to AVT" decisions
import { useState } from "react";
import { discoveryApi, type DiscoveryReport } from "@/services/api";
import ContinuousResearchPanel from "@/components/discovery/ContinuousResearchPanel";

// ── Color helpers ───────────────────────────────────────────────
const wrColor = (wr: number) =>
  wr >= 0.65 ? "#10B981" : wr >= 0.55 ? "#F59E0B" : wr >= 0.45 ? "#F97316" : "#EF4444";
const rColor = (r: number) =>
  r >= 2.0 ? "#10B981" : r >= 1.0 ? "#F59E0B" : r >= 0 ? "#F97316" : "#EF4444";
const confColor = (c: number) =>
  c >= 0.7 ? "#10B981" : c >= 0.4 ? "#F59E0B" : "#EF4444";

const CARD = "bg-[#131825] border border-[#1E2740] rounded-xl p-4";
const BADGE = "inline-block text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full";

// ── Stages display ──────────────────────────────────────────────
const STAGES = [
  { key: "parsing",     label: "Parse Intent",     icon: "1" },
  { key: "extracting",  label: "Extract Pattern",  icon: "2" },
  { key: "scanning",    label: "Scan Universe",    icon: "3" },
  { key: "backtesting", label: "Backtest",         icon: "4" },
  { key: "correlating", label: "Regime Analysis",  icon: "5" },
  { key: "generating",  label: "Generate DNA",     icon: "6" },
  { key: "review",      label: "Review",           icon: "7" },
];

const STATUS_ORDER = ["parsing", "extracting", "scanning", "backtesting", "correlating", "generating", "review"];

function StageIndicator({ currentStatus }: { currentStatus: string }) {
  const idx = STATUS_ORDER.indexOf(currentStatus);
  return (
    <div className="flex items-center gap-1 mb-6">
      {STAGES.map((s, i) => {
        const done = i < idx;
        const active = i === idx;
        const bg = done ? "bg-[#10B981]" : active ? "bg-[#8B5CF6]" : "bg-[#1E2740]";
        const text = done || active ? "text-white" : "text-[#475569]";
        return (
          <div key={s.key} className="flex items-center">
            <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${bg} ${text}`}>
              {done ? "\u2713" : s.icon}
            </div>
            <span className={`text-[10px] ml-1 mr-2 hidden sm:inline ${active ? "text-[#C4B5FD]" : "text-[#475569]"}`}>
              {s.label}
            </span>
            {i < STAGES.length - 1 && (
              <div className={`w-4 h-[2px] ${done ? "bg-[#10B981]" : "bg-[#1E2740]"}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Main page ───────────────────────────────────────────────────
export default function PatternDiscoveryPage() {
  const [mode, setMode] = useState<"oneshot" | "continuous">("oneshot");
  const [input, setInput] = useState("");
  const [threshold, setThreshold] = useState(90);
  const [lookback, setLookback] = useState(3);
  const [scope, setScope] = useState("fno");
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<DiscoveryReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"overview" | "stocks" | "regime" | "time" | "strategy">("overview");
  const [acceptNotes, setAcceptNotes] = useState("");
  const [rejectReason, setRejectReason] = useState("");
  const [actionDone, setActionDone] = useState<string | null>(null);

  const run = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setError(null);
    setReport(null);
    setActionDone(null);
    setTab("overview");
    try {
      const data = await discoveryApi.discover(input, {
        similarity_threshold: threshold / 100,
        lookback_years: lookback,
        scope,
      });
      setReport(data);
    } catch (e: any) {
      setError(e.message || "Discovery failed");
    } finally {
      setLoading(false);
    }
  };

  const handleAccept = async () => {
    if (!report) return;
    try {
      await discoveryApi.accept(report.discovery_id, acceptNotes);
      setActionDone("accepted");
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleReject = async () => {
    if (!report) return;
    try {
      await discoveryApi.reject(report.discovery_id, rejectReason);
      setActionDone("rejected");
    } catch (e: any) {
      setError(e.message);
    }
  };

  return (
    <div className="min-h-screen bg-[#0A0E1A] text-[#F1F5F9] p-4 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-4">
        <h1 className="text-xl font-bold text-[#C4B5FD] tracking-tight">Pattern Discovery Engine</h1>
        <p className="text-xs text-[#64748B] mt-1">
          {mode === "oneshot"
            ? "Describe a market move in natural language. PTIS will find similar patterns, backtest them, and generate a strategy."
            : "Background loop tests many (symbol × pattern) combos. Validated strategies appear as drafts that you review and send to AVT individually. No real-money trading."}
        </p>
      </div>

      {/* Mode toggle */}
      <div className="mb-4 flex gap-1 p-1 bg-[#0F141F] border border-[#1E2740] rounded-lg w-fit">
        <button
          onClick={() => setMode("oneshot")}
          className={`text-xs px-3 py-1.5 rounded transition-colors ${
            mode === "oneshot"
              ? "bg-[#8B5CF6] text-white font-semibold"
              : "text-[#94A3B8] hover:text-[#F1F5F9]"
          }`}
        >
          One-Shot Discovery
        </button>
        <button
          onClick={() => setMode("continuous")}
          className={`text-xs px-3 py-1.5 rounded transition-colors ${
            mode === "continuous"
              ? "bg-[#8B5CF6] text-white font-semibold"
              : "text-[#94A3B8] hover:text-[#F1F5F9]"
          }`}
        >
          Continuous Research
        </button>
      </div>

      {/* Continuous mode renders separate panel */}
      {mode === "continuous" && <ContinuousResearchPanel />}

      {/* One-shot mode UI (original) */}
      {mode === "oneshot" && (<>

      {/* Input area */}
      <div className={CARD + " mb-4"}>
        <label className="block text-[10px] text-[#64748B] mb-2 uppercase tracking-wider">
          Describe what you saw
        </label>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder='e.g. "Look at Reliance 15-min chart, today breakout at 10:30 with solid momentum and volume surge. Find similar patterns across FNO stocks."'
          rows={3}
          className="w-full bg-[#0A0E1A] border border-[#2A3050] rounded-lg px-3 py-2 text-sm text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6] transition-colors resize-none"
        />
        {/* Controls row */}
        <div className="flex flex-wrap gap-3 mt-3 items-end">
          <div>
            <label className="block text-[10px] text-[#64748B] mb-1">Similarity</label>
            <div className="flex items-center gap-2">
              <input
                type="range" min={70} max={99} value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-24 accent-[#8B5CF6]"
              />
              <span className="text-xs text-[#C4B5FD] font-mono w-8">{threshold}%</span>
            </div>
          </div>
          <div>
            <label className="block text-[10px] text-[#64748B] mb-1">Lookback</label>
            <select
              value={lookback} onChange={(e) => setLookback(Number(e.target.value))}
              className="bg-[#0A0E1A] border border-[#2A3050] rounded px-2 py-1 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#8B5CF6]"
            >
              {[1, 2, 3, 5].map((y) => <option key={y} value={y}>{y} year{y > 1 ? "s" : ""}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-[10px] text-[#64748B] mb-1">Scope</label>
            <select
              value={scope} onChange={(e) => setScope(e.target.value)}
              className="bg-[#0A0E1A] border border-[#2A3050] rounded px-2 py-1 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#8B5CF6]"
            >
              <option value="fno">FNO Universe (~200)</option>
              <option value="nifty50">Nifty 50</option>
            </select>
          </div>
          <button
            onClick={run}
            disabled={loading || !input.trim()}
            className="ml-auto bg-[#8B5CF6] hover:bg-[#7C3AED] disabled:opacity-40 text-white text-sm font-semibold px-6 py-2 rounded-lg transition-colors"
          >
            {loading ? "Discovering..." : "Discover Pattern"}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-[#1C1016] border border-[#EF4444]/30 rounded-lg px-4 py-3 mb-4 text-sm text-[#FCA5A5]">
          {error}
        </div>
      )}

      {/* Loading with stage indicator */}
      {loading && (
        <div className={CARD + " mb-4"}>
          <StageIndicator currentStatus="scanning" />
          <div className="flex items-center gap-3 text-sm text-[#94A3B8]">
            <div className="w-4 h-4 border-2 border-[#8B5CF6] border-t-transparent rounded-full animate-spin" />
            Running 7-stage discovery pipeline...
          </div>
        </div>
      )}

      {/* Report */}
      {report && !loading && (
        <>
          {/* Stage indicator */}
          <StageIndicator currentStatus={report.status} />

          {/* Tabs */}
          <div className="flex gap-1 mb-4 border-b border-[#1E2740]">
            {(["overview", "stocks", "regime", "time", "strategy"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-2 text-xs font-medium uppercase tracking-wider transition-colors ${
                  tab === t
                    ? "text-[#C4B5FD] border-b-2 border-[#8B5CF6]"
                    : "text-[#64748B] hover:text-[#94A3B8]"
                }`}
              >
                {t}
              </button>
            ))}
          </div>

          {/* Tab content */}
          {tab === "overview" && <OverviewTab report={report} />}
          {tab === "stocks" && <StocksTab report={report} />}
          {tab === "regime" && <RegimeTab report={report} />}
          {tab === "time" && <TimeTab report={report} />}
          {tab === "strategy" && <StrategyTab report={report} />}

          {/* Accept / Reject */}
          {report.status === "review" && !actionDone && (
            <div className={CARD + " mt-4"}>
              <h3 className="text-sm font-semibold text-[#C4B5FD] mb-3">Review Decision</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Accept */}
                <div className="space-y-2">
                  <textarea
                    value={acceptNotes}
                    onChange={(e) => setAcceptNotes(e.target.value)}
                    placeholder="Notes (optional)"
                    rows={2}
                    className="w-full bg-[#0A0E1A] border border-[#2A3050] rounded px-3 py-2 text-xs text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#10B981] resize-none"
                  />
                  <button
                    onClick={handleAccept}
                    className="w-full bg-[#10B981] hover:bg-[#059669] text-white text-sm font-semibold py-2 rounded-lg transition-colors"
                  >
                    Accept & Save Strategy
                  </button>
                </div>
                {/* Reject */}
                <div className="space-y-2">
                  <textarea
                    value={rejectReason}
                    onChange={(e) => setRejectReason(e.target.value)}
                    placeholder="Reason (optional)"
                    rows={2}
                    className="w-full bg-[#0A0E1A] border border-[#2A3050] rounded px-3 py-2 text-xs text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#EF4444] resize-none"
                  />
                  <button
                    onClick={handleReject}
                    className="w-full bg-[#1C1016] border border-[#EF4444]/40 hover:bg-[#EF4444]/10 text-[#FCA5A5] text-sm font-semibold py-2 rounded-lg transition-colors"
                  >
                    Reject
                  </button>
                </div>
              </div>
            </div>
          )}
          {actionDone && (
            <div className={`${CARD} mt-4 border-${actionDone === "accepted" ? "[#10B981]" : "[#EF4444]"}/30`}>
              <p className="text-sm font-semibold" style={{ color: actionDone === "accepted" ? "#10B981" : "#EF4444" }}>
                Strategy {actionDone === "accepted" ? "accepted and saved to strategy library!" : "rejected."}
              </p>
            </div>
          )}
        </>
      )}

      </>)}{/* end oneshot mode */}
    </div>
  );
}

// ── Overview Tab ────────────────────────────────────────────────
function OverviewTab({ report }: { report: DiscoveryReport }) {
  const q = report.query;
  return (
    <div className="space-y-4">
      {/* Query summary */}
      {q && (
        <div className={CARD}>
          <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider mb-2">Query Parsed</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
            <div><span className="text-[#64748B] text-xs">Symbol</span><br/><span className="font-mono text-[#C4B5FD]">{q.symbol}</span></div>
            <div><span className="text-[#64748B] text-xs">Timeframe</span><br/><span className="font-mono">{q.timeframe}</span></div>
            <div><span className="text-[#64748B] text-xs">Pattern</span><br/><span className="font-mono">{q.pattern_type.replace(/_/g, " ")}</span></div>
            <div><span className="text-[#64748B] text-xs">Time</span><br/><span className="font-mono">{q.event_time || "auto"}</span></div>
          </div>
          {q.description && <p className="text-xs text-[#94A3B8] mt-2">{q.description}</p>}
        </div>
      )}

      {/* Key stats */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
        <StatCard label="Matches" value={report.total_matches.toString()} />
        <StatCard label="Win Rate" value={`${(report.overall_win_rate * 100).toFixed(1)}%`} color={wrColor(report.overall_win_rate)} />
        <StatCard label="Avg R" value={report.overall_avg_r.toFixed(2)} color={rColor(report.overall_avg_r)} />
        <StatCard label="Profit Factor" value={report.overall_profit_factor.toFixed(1)} color={report.overall_profit_factor >= 2 ? "#10B981" : "#F59E0B"} />
        <StatCard label="Hold Time" value={`${report.median_hold_minutes}m`} />
        <StatCard label="Avg MAE" value={`${report.avg_mae.toFixed(2)}R`} />
      </div>

      {/* Fingerprint radar */}
      {report.fingerprint && (
        <div className={CARD}>
          <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider mb-3">Pattern Fingerprint</h3>
          <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
            {report.fingerprint.labels.map((label, i) => (
              <div key={label} className="text-center">
                <div className="h-16 flex items-end justify-center">
                  <div
                    className="w-6 rounded-t"
                    style={{
                      height: `${Math.max(report.fingerprint!.vector[i] * 100, 4)}%`,
                      backgroundColor: `hsl(${260 - report.fingerprint!.vector[i] * 120}, 70%, 60%)`,
                    }}
                  />
                </div>
                <span className="text-[9px] text-[#64748B] leading-tight block mt-1">
                  {label.replace(/_/g, " ")}
                </span>
                <span className="text-[10px] font-mono text-[#94A3B8]">
                  {report.fingerprint!.vector[i].toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Quick regime + time summary */}
      {report.correlation && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className={CARD}>
            <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider mb-2">Best Regime</h3>
            <p className="text-lg font-bold text-[#10B981]">{report.correlation.best_regime || "N/A"}</p>
            <p className="text-xs text-[#64748B] mt-1">Worst: <span className="text-[#EF4444]">{report.correlation.worst_regime || "N/A"}</span></p>
          </div>
          <div className={CARD}>
            <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider mb-2">Golden Window</h3>
            <p className="text-lg font-bold text-[#F59E0B]">{report.correlation.best_time_window || "N/A"}</p>
            <p className="text-xs text-[#64748B] mt-1">IST time slot with highest win rate</p>
          </div>
        </div>
      )}

      {report.error && (
        <div className="bg-[#1C1016] border border-[#F59E0B]/30 rounded-lg px-4 py-3 text-xs text-[#FCD34D]">
          {report.error}
        </div>
      )}
    </div>
  );
}

// ── Stocks Tab ──────────────────────────────────────────────────
function StocksTab({ report }: { report: DiscoveryReport }) {
  return (
    <div className={CARD}>
      <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider mb-3">
        Top Matching Stocks ({report.top_stocks.length})
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-[#64748B] border-b border-[#1E2740]">
              <th className="text-left py-2 pr-4">Symbol</th>
              <th className="text-right py-2 px-3">Instances</th>
              <th className="text-right py-2 px-3">Win Rate</th>
              <th className="text-right py-2 px-3">Avg R</th>
              <th className="text-left py-2 pl-3">Bar</th>
            </tr>
          </thead>
          <tbody>
            {report.top_stocks.map((s) => (
              <tr key={s.symbol} className="border-b border-[#1E2740]/50 hover:bg-[#1E2740]/30">
                <td className="py-2 pr-4 font-mono text-[#C4B5FD] font-semibold">{s.symbol}</td>
                <td className="py-2 px-3 text-right">{s.instances}</td>
                <td className="py-2 px-3 text-right font-mono" style={{ color: wrColor(s.win_rate) }}>
                  {(s.win_rate * 100).toFixed(0)}%
                </td>
                <td className="py-2 px-3 text-right font-mono" style={{ color: rColor(s.avg_r) }}>
                  {s.avg_r.toFixed(1)}
                </td>
                <td className="py-2 pl-3">
                  <div className="w-24 h-2 bg-[#1E2740] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.min(s.win_rate * 100, 100)}%`,
                        backgroundColor: wrColor(s.win_rate),
                      }}
                    />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Regime Tab ──────────────────────────────────────────────────
function RegimeTab({ report }: { report: DiscoveryReport }) {
  const regimes = report.correlation?.regime_breakdown || [];
  return (
    <div className="space-y-4">
      {regimes.map((rb) => (
        <div key={rb.regime} className={CARD}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-[#F1F5F9]">{rb.regime.replace(/_/g, " ").toUpperCase()}</span>
              <span className={BADGE} style={{
                backgroundColor: wrColor(rb.win_rate) + "20",
                color: wrColor(rb.win_rate),
              }}>
                {rb.win_rate >= 0.65 ? "FULL SIZE" : rb.win_rate >= 0.55 ? "STANDARD" : rb.win_rate >= 0.45 ? "REDUCE" : "SKIP"}
              </span>
            </div>
            <span className="text-xs text-[#64748B]">N={rb.count}</span>
          </div>
          <div className="grid grid-cols-4 gap-3 text-xs mb-3">
            <div>
              <span className="text-[#64748B]">Win Rate</span>
              <p className="font-mono font-bold" style={{ color: wrColor(rb.win_rate) }}>{(rb.win_rate * 100).toFixed(0)}%</p>
            </div>
            <div>
              <span className="text-[#64748B]">Avg R</span>
              <p className="font-mono font-bold" style={{ color: rColor(rb.avg_r) }}>{rb.avg_r.toFixed(1)}</p>
            </div>
            <div>
              <span className="text-[#64748B]">PF</span>
              <p className="font-mono font-bold">{rb.profit_factor.toFixed(1)}</p>
            </div>
            <div>
              <span className="text-[#64748B]">Best/Worst R</span>
              <p className="font-mono text-[10px]">
                <span className="text-[#10B981]">+{rb.avg_r > 0 ? rb.avg_r.toFixed(1) : "0"}</span>
                {" / "}
                <span className="text-[#EF4444]">{rb.avg_r < 0 ? rb.avg_r.toFixed(1) : "-1.0"}</span>
              </p>
            </div>
          </div>
          {rb.why_works && (
            <div className="bg-[#0A1A14] border-l-2 border-[#10B981] px-3 py-2 text-xs text-[#86EFAC] mb-2">
              <span className="font-semibold">WHY IT WORKS: </span>{rb.why_works}
            </div>
          )}
          {rb.why_fails && (
            <div className="bg-[#1C1016] border-l-2 border-[#EF4444] px-3 py-2 text-xs text-[#FCA5A5]">
              <span className="font-semibold">WHY IT FAILS: </span>{rb.why_fails}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ── Time Tab ────────────────────────────────────────────────────
function TimeTab({ report }: { report: DiscoveryReport }) {
  const slots = report.correlation?.time_breakdown || [];
  const maxCount = Math.max(...slots.map((s) => s.count), 1);
  return (
    <div className={CARD}>
      <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider mb-4">
        Time-of-Day Performance (IST)
      </h3>
      <div className="space-y-3">
        {slots.map((s) => (
          <div key={s.slot}>
            <div className="flex items-center justify-between text-xs mb-1">
              <div className="flex items-center gap-2">
                <span className="font-mono text-[#F1F5F9] w-24">{s.slot}</span>
                <span className="text-[#64748B]">{s.label}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="font-mono" style={{ color: wrColor(s.win_rate) }}>
                  {s.count > 0 ? `${(s.win_rate * 100).toFixed(0)}%` : "—"}
                </span>
                <span className="text-[#64748B] w-12 text-right">N={s.count}</span>
              </div>
            </div>
            <div className="w-full h-3 bg-[#1E2740] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${(s.count / maxCount) * 100}%`,
                  backgroundColor: s.count > 0 ? wrColor(s.win_rate) : "#1E2740",
                  opacity: s.count >= 5 ? 1 : 0.4,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Strategy Tab ────────────────────────────────────────────────
function StrategyTab({ report }: { report: DiscoveryReport }) {
  const gs = report.generated_strategy;
  if (!gs) return <p className="text-sm text-[#64748B]">No strategy generated yet.</p>;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className={CARD}>
        <div className="flex items-center justify-between mb-2">
          <div>
            <h3 className="text-sm font-bold text-[#C4B5FD]">{gs.display_name}</h3>
            <span className="text-xs text-[#64748B]">{gs.short_code} | {gs.category}</span>
          </div>
          <div className="text-right">
            <span className="text-xs text-[#64748B]">Confidence</span>
            <p className="text-lg font-bold font-mono" style={{ color: confColor(gs.confidence) }}>
              {(gs.confidence * 100).toFixed(0)}%
            </p>
          </div>
        </div>
        <p className="text-xs text-[#94A3B8]">{gs.llm_reasoning}</p>
      </div>

      {/* YAML */}
      <div className={CARD}>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider">Strategy DNA (YAML)</h3>
          <button
            onClick={() => navigator.clipboard.writeText(gs.yaml_content)}
            className="text-[10px] text-[#8B5CF6] hover:text-[#C4B5FD] transition-colors"
          >
            Copy
          </button>
        </div>
        <pre className="bg-[#0A0E1A] border border-[#1E2740] rounded-lg p-3 text-[11px] text-[#94A3B8] font-mono overflow-x-auto max-h-[500px] overflow-y-auto whitespace-pre-wrap">
          {gs.yaml_content}
        </pre>
      </div>
    </div>
  );
}

// ── Stat card helper ────────────────────────────────────────────
function StatCard({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className={CARD + " text-center"}>
      <span className="text-[10px] text-[#64748B] uppercase tracking-wider">{label}</span>
      <p className="text-lg font-bold font-mono mt-1" style={{ color: color || "#F1F5F9" }}>
        {value}
      </p>
    </div>
  );
}
