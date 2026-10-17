// src/pages/AVTPage.tsx
// Unified AVT console — Dashboard, Competition, Improvements, Scan Logs in one page
import { useCallback, useEffect, useRef, useState } from "react";
import { avtApi, type AVTHealth, type ReplayJobStatus, type CapitalSleeve } from "@/services/api";
import CapitalSleevesConfig from "@/components/avt/CapitalSleevesConfig";
import {
  LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip,
  CartesianGrid, Legend,
} from "recharts";

const CARD = "bg-[#1A2035] border border-[#2A3050] rounded-lg p-4";
const WS_URL = import.meta.env.VITE_API_URL
  ? import.meta.env.VITE_API_URL.replace(/^http/, "ws") + "/ws/session"
  : `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/session`;

type Tab = "dashboard" | "chat" | "competition" | "improvements" | "activity" | "health";

export default function AVTPage() {
  const [tab, setTab] = useState<Tab>("dashboard");
  const [proactiveMsg, setProactiveMsg] = useState<string | null>(null);

  // Check for proactive messages every 30s
  useEffect(() => {
    const check = () => {
      avtApi.proactiveMessage().then((r: any) => {
        if (r?.has_message && r.message) setProactiveMsg(r.message);
      }).catch(() => {});
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col h-[calc(100vh-56px)]">
      {/* Header */}
      <header className="px-4 py-3 border-b border-[#2A3050] shrink-0">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-sm font-bold text-[#F1F5F9]">Autonomous Virtual Trader</h1>
            <p className="text-[10px] text-[#64748B]">Your AI trading partner — observes, learns, proves, communicates</p>
          </div>
          {proactiveMsg && (
            <div className="flex items-center gap-2 bg-[#8B5CF6]/10 border border-[#8B5CF6]/30 rounded-lg px-3 py-1.5 max-w-md">
              <span className="text-[10px] text-[#C4B5FD] line-clamp-2">{proactiveMsg}</span>
              <button onClick={() => { setProactiveMsg(null); setTab("chat"); }}
                className="text-[9px] text-[#8B5CF6] hover:text-[#A78BFA] shrink-0">View</button>
            </div>
          )}
        </div>
      </header>

      {/* Tabs */}
      <div className="px-4 py-2 border-b border-[#2A3050] shrink-0 flex gap-1">
        {([
          { key: "dashboard", label: "Dashboard" },
          { key: "chat", label: "Talk to AVT" },
          { key: "competition", label: "AVT vs You" },
          { key: "improvements", label: "Improvements" },
          { key: "activity", label: "Activity" },
          { key: "health", label: "Strategy Health" },
        ] as const).map((t) => (
          <button key={t.key} onClick={() => setTab(t.key)}
            className={`px-4 py-2 text-xs font-medium uppercase tracking-wider transition-colors ${
              tab === t.key ? "text-[#C4B5FD] border-b-2 border-[#8B5CF6]" : "text-[#64748B] hover:text-[#94A3B8]"
            }`}>{t.label}</button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {tab === "dashboard" && <DashboardTab />}
        {tab === "chat" && <ChatTab />}
        {tab === "competition" && <CompetitionTab />}
        {tab === "improvements" && <ImprovementsTab />}
        {tab === "activity" && <ActivityTab />}
        {tab === "health" && <StrategyHealthTab />}
      </div>
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════
//  DASHBOARD TAB
// ══════════════════════════════════════════════════════════════════

function DashboardTab() {
  const [status, setStatus] = useState<any>(null);
  const [health, setHealth] = useState<AVTHealth | null>(null);
  const [openTrades, setOpenTrades] = useState<any[]>([]);
  const [closedTrades, setClosedTrades] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const [s, h, open, closed] = await Promise.all([
        avtApi.status(),
        avtApi.health().catch(() => null),
        avtApi.openTrades(),
        avtApi.trades({ status: "closed", limit: 100 }),
      ]);
      if (!mountedRef.current) return;
      setStatus(s); setHealth(h); setOpenTrades(open.trades); setClosedTrades(closed.trades);
      setError(null);
    } catch (e: any) { if (mountedRef.current) setError(e.message); }
    finally { if (mountedRef.current) setLoading(false); }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    refresh();
    const iv = setInterval(refresh, 30_000);
    return () => { mountedRef.current = false; clearInterval(iv); };
  }, [refresh]);

  // WebSocket live
  useEffect(() => {
    let ws: WebSocket | null = null;
    try {
      const token = localStorage.getItem("ptis_access_token");
      ws = new WebSocket(token ? `${WS_URL}?token=${token}` : WS_URL);
      ws.onmessage = (evt) => {
        const msg = JSON.parse(evt.data);
        if (msg.type?.startsWith("avt.")) refresh();
      };
    } catch {}
    return () => ws?.close();
  }, [refresh]);

  const handleStart = async () => { try { await avtApi.start(); refresh(); } catch (e: any) { setError(e.message); } };
  const handleStop = async () => { try { await avtApi.stop(); refresh(); } catch (e: any) { setError(e.message); } };

  // Benchmark replay — build the strategy baseline (edge + per-stock ranking).
  const [replay, setReplay] = useState<ReplayJobStatus | null>(null);
  useEffect(() => { avtApi.replayStatus().then(setReplay).catch(() => {}); }, []);
  useEffect(() => {
    if (replay?.state !== "running") return;
    const id = setInterval(() => { avtApi.replayStatus().then(setReplay).catch(() => {}); }, 4000);
    return () => clearInterval(id);
  }, [replay?.state]);
  const handleBuildBaseline = async () => {
    try { setReplay(await avtApi.startReplay()); } catch (e: any) { setError(e.message); }
  };

  if (loading) return <Loading />;
  if (!status) {
    // Not started yet (status 503s until POST /avt/start). Show a Start button rather than a dead-end error.
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-24 text-center">
        <div className="text-[#94A3B8]">Your AVT isn&apos;t running yet.</div>
        <div className="text-xs text-[#64748B] max-w-md">
          Starting it creates your engine and loads any activated strategy templates (e.g. Cascade), then scans,
          ranks and takes virtual trades.
        </div>
        <button
          onClick={handleStart}
          className="px-5 py-2 text-sm font-medium bg-[#10B981]/20 text-[#10B981] rounded-lg hover:bg-[#10B981]/30 transition-colors"
        >
          ▶ Start AVT
        </button>
        {error && <div className="text-[10px] text-[#EF4444]">{error}</div>}
      </div>
    );
  }

  const isRunning = status?.status === "running" || status?.status === "scanning";

  const kpis = [
    { label: "Status", value: status?.status?.toUpperCase() || "—", color: isRunning ? "#10B981" : "#64748B" },
    { label: "Open", value: status?.open_positions ?? 0, color: "#3B82F6" },
    { label: "Trades", value: status?.total_trades ?? 0, color: "#A78BFA" },
    { label: "Equity", value: `${((status?.portfolio_equity ?? 0) / 100000).toFixed(1)}L`, color: "#F1F5F9" },
    { label: "Drawdown", value: `${(status?.portfolio_drawdown_pct ?? 0).toFixed(1)}%`, color: (status?.portfolio_drawdown_pct ?? 0) > 5 ? "#EF4444" : "#10B981" },
    { label: "Cycle", value: status?.current_cycle ?? 0, color: "#64748B" },
  ];

  const equityCurve = closedTrades.slice().reverse().reduce((acc: any[], t: any, i: number) => {
    const prev = acc.length > 0 ? acc[acc.length - 1].equity : 0;
    acc.push({ idx: i + 1, equity: prev + (t.pnl_r ?? 0) });
    return acc;
  }, []);

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex justify-between items-center">
        <div className="flex gap-2 items-center">
          <span className={`inline-block w-2 h-2 rounded-full ${isRunning ? "bg-[#10B981] animate-pulse" : "bg-[#64748B]"}`} />
          <span className="text-xs text-[#94A3B8] uppercase">{status?.status}</span>
        </div>
        <div className="flex gap-2 items-center">
          <button
            onClick={handleBuildBaseline}
            disabled={replay?.state === "running"}
            title="Replay your taught strategies over historical data to build the edge + stock-ranking baseline (benchmark, frozen v1.0)."
            className="px-3 py-1 text-xs bg-[#8B5CF6]/20 text-[#C4B5FD] rounded hover:bg-[#8B5CF6]/30 transition-colors disabled:opacity-50"
          >
            {replay?.state === "running" ? "Building baseline…" : "Build baseline"}
          </button>
          {isRunning ? (
            <button onClick={handleStop} className="px-3 py-1 text-xs bg-[#EF4444]/20 text-[#EF4444] rounded hover:bg-[#EF4444]/30 transition-colors">Stop AVT</button>
          ) : (
            <button onClick={handleStart} className="px-3 py-1 text-xs bg-[#10B981]/20 text-[#10B981] rounded hover:bg-[#10B981]/30 transition-colors">Start AVT</button>
          )}
        </div>
      </div>

      {/* Benchmark-replay progress (build baseline) */}
      {replay && replay.state !== "idle" && (
        <div className="text-[11px] flex items-center gap-2 px-3 py-2 rounded border bg-[#1A2035]"
          style={{ borderColor: replay.state === "error" ? "#EF4444" : replay.state === "done" ? "#10B981" : "#8B5CF6" }}>
          <span className="uppercase tracking-wider font-bold"
            style={{ color: replay.state === "error" ? "#EF4444" : replay.state === "done" ? "#10B981" : "#C4B5FD" }}>
            {replay.state === "running" ? "Building baseline" : replay.state}
          </span>
          <span className="text-[#94A3B8]">{replay.message}</span>
        </div>
      )}

      {/* Health banner + funnel — "is AVT working?" */}
      {health && (
        <div
          className={CARD + " space-y-2"}
          style={{ borderColor: health.health === "healthy" ? "#10B981" : health.health === "quiet" ? "#F59E0B" : "#EF4444" }}
        >
          <div className="flex items-center gap-2">
            <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: health.health === "healthy" ? "#10B981" : health.health === "quiet" ? "#F59E0B" : "#EF4444" }} />
            <span className="text-xs font-bold uppercase tracking-wider" style={{ color: health.health === "healthy" ? "#10B981" : health.health === "quiet" ? "#F59E0B" : "#EF4444" }}>
              {health.health === "healthy" ? "Healthy" : health.health === "quiet" ? "Quiet" : "Not Ready"}
            </span>
            <span className="text-xs text-[#94A3B8]">{health.reason}</span>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-[#64748B]">
            <span>{health.strategies_with_scoring_config} tradeable strategies</span>
            <span>{health.watchlist_size} symbols</span>
            <span>feed: {health.market_data_source}{health.market_open ? " · open" : " · closed"}</span>
            <span>last scan: {health.last_scan_at ? new Date(health.last_scan_at).toLocaleTimeString() : "—"}</span>
          </div>
          {health.last_cycle && (
            <div className="flex items-center gap-1 text-[10px] font-num text-[#94A3B8] flex-wrap">
              <span>{health.last_cycle.scanned} scanned</span><span className="text-[#475569]">→</span>
              <span>{health.last_cycle.candidates} candidates</span><span className="text-[#475569]">→</span>
              <span>{health.last_cycle.ranked} ranked</span><span className="text-[#475569]">→</span>
              <span>{health.last_cycle.filtered} eligible</span><span className="text-[#475569]">→</span>
              <span className="text-[#10B981]">{health.last_cycle.executed} traded</span>
              {health.last_cycle.executed === 0 && health.dominant_skip_reason && (
                <span className="text-[#F59E0B]">({health.dominant_skip_reason})</span>
              )}
            </div>
          )}
          {health.recent_errors.length > 0 && (
            <div className="text-[10px] text-[#EF4444]">⚠ {health.recent_errors[0]}</div>
          )}
        </div>
      )}

      {/* KPIs */}
      <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
        {kpis.map((k) => (
          <div key={k.label} className={CARD + " text-center py-3"}>
            <div className="text-[9px] uppercase tracking-widest text-[#64748B] mb-1">{k.label}</div>
            <div className="text-lg font-bold font-num" style={{ color: k.color }}>{k.value}</div>
          </div>
        ))}
      </div>

      {/* Capital sleeves — separate pools per instrument (futures / options / equity run concurrently) */}
      {status?.partitioned && (status?.capital_sleeves?.length ?? 0) > 0 && (
        <div className={CARD}>
          <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider mb-3">Capital sleeves</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {status!.capital_sleeves.map((s: CapitalSleeve) => {
              const usedPct = s.capital > 0 ? Math.min(100, (s.deployed / s.capital) * 100) : 0;
              return (
                <div key={s.name} className="bg-[#141a2e] border border-[#2A3050] rounded p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-bold text-[#E2E8F0] uppercase tracking-wide">{s.name}</span>
                    <span className="text-[9px] text-[#64748B]">{s.instruments.join(" · ")}</span>
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-lg font-bold font-num text-[#F1F5F9]">₹{(s.equity / 100000).toFixed(2)}L</span>
                    <span className="text-[10px] text-[#64748B]">of ₹{(s.capital / 100000).toFixed(2)}L</span>
                  </div>
                  <div className="h-1.5 bg-[#0B0F1E] rounded mt-1.5 mb-1 overflow-hidden">
                    <div className="h-full bg-[#3B82F6]" style={{ width: `${usedPct}%` }} />
                  </div>
                  <div className="flex items-center justify-between text-[10px] text-[#94A3B8] font-num">
                    <span>free ₹{(s.available / 100000).toFixed(2)}L</span>
                    <span>{s.open_positions}/{s.max_positions} slots</span>
                    <span style={{ color: s.drawdown_pct > 5 ? "#EF4444" : "#64748B" }}>dd {s.drawdown_pct.toFixed(1)}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Capital sleeves editor — set your account split (their money decision) */}
      <CapitalSleevesConfig />

      {/* Equity curve */}
      {equityCurve.length > 3 && (
        <div className={CARD}>
          <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider mb-3">Cumulative R</h3>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={equityCurve}>
              <CartesianGrid stroke="#2A3050" strokeDasharray="3 3" />
              <XAxis dataKey="idx" tick={{ fill: "#64748B", fontSize: 10 }} />
              <YAxis tick={{ fill: "#64748B", fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#1A2035", border: "1px solid #2A3050", borderRadius: 8, fontSize: 12 }} />
              <Line type="monotone" dataKey="equity" stroke="#8B5CF6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Open positions */}
      {openTrades.length > 0 && (
        <div className={CARD}>
          <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider mb-2">Open Positions</h3>
          {openTrades.map((t: any) => (
            <div key={t.id} className="flex justify-between items-center text-xs py-1.5 border-b border-[#2A3050]/50 last:border-0">
              <span className="text-[#F1F5F9] font-medium w-28 truncate" title={t.instrument === "option" ? t.tradeable : t.symbol}>
                {t.symbol}
                {t.instrument === "option" && (
                  <span className="ml-1 text-[9px] text-[#A78BFA]">{t.option_right} {t.option_strike}</span>
                )}
              </span>
              <span className={`uppercase text-[10px] w-10 ${t.direction === "long" ? "text-[#10B981]" : "text-[#EF4444]"}`}>{t.direction}</span>
              <span className="text-[#94A3B8] font-num" title={t.instrument === "option" ? "entry premium" : "entry price"}>
                {t.entry_price.toFixed(2)}{t.instrument === "option" && <span className="text-[9px] text-[#64748B]"> ×{t.option_lots}</span>}
              </span>
              <span className="text-[#10B981] font-num">MFE {t.max_favorable_r.toFixed(2)}R</span>
              <span className="text-[#8B5CF6] font-num">#{t.ranking_position}</span>
            </div>
          ))}
        </div>
      )}

      {/* Recent closed */}
      {closedTrades.length > 0 && (
        <div className={CARD}>
          <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider mb-2">Recent Trades</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-[10px] text-[#64748B] uppercase tracking-wider border-b border-[#2A3050]">
                  <th className="text-left py-2">Symbol</th><th className="text-left py-2">Dir</th>
                  <th className="text-right py-2">R</th><th className="text-right py-2">PnL</th>
                  <th className="text-left py-2">Exit</th><th className="text-left py-2">Date</th>
                </tr>
              </thead>
              <tbody>
                {closedTrades.slice(0, 20).map((t) => (
                  <tr key={t.id} className="border-b border-[#2A3050]/30">
                    <td className="py-1.5 text-[#F1F5F9]" title={t.instrument === "option" ? t.tradeable : t.symbol}>
                      {t.symbol}
                      {t.instrument === "option" && (
                        <span className="ml-1 text-[9px] text-[#A78BFA]">{t.option_right} {t.option_strike}</span>
                      )}
                    </td>
                    <td className={t.direction === "long" ? "text-[#10B981]" : "text-[#EF4444]"}>{t.direction}</td>
                    <td className={`text-right font-num font-bold ${(t.pnl_r ?? 0) >= 0 ? "text-[#10B981]" : "text-[#EF4444]"}`}>
                      {t.pnl_r != null ? `${t.pnl_r >= 0 ? "+" : ""}${t.pnl_r.toFixed(2)}` : "—"}
                    </td>
                    <td className={`text-right font-num ${(t.pnl_absolute ?? 0) >= 0 ? "text-[#10B981]" : "text-[#EF4444]"}`}>
                      {t.pnl_absolute != null ? `${t.pnl_absolute >= 0 ? "+" : ""}${t.pnl_absolute.toFixed(0)}` : "—"}
                    </td>
                    <td className="text-[#64748B]">{t.exit_type || "—"}</td>
                    <td className="text-[#64748B]">{t.exit_time?.slice(0, 10) || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════
//  COMPETITION TAB
// ══════════════════════════════════════════════════════════════════

function CompetitionTab() {
  const [strategies, setStrategies] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [report, setReport] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [versions, setVersions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [periodDays, setPeriodDays] = useState(30);

  useEffect(() => {
    avtApi.status().then((s) => {
      setStrategies(s.active_strategies || []);
      if (s.active_strategies?.length) setSelected(s.active_strategies[0]);
    }).catch(() => {}).finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setLoading(true);
    Promise.all([
      avtApi.competition(selected, periodDays).catch(() => null),
      avtApi.competitionHistory(selected, 30).catch(() => ({ snapshots: [] })),
      avtApi.versions(selected).catch(() => ({ versions: [] })),
    ]).then(([r, h, v]) => {
      setReport(r); setHistory(h?.snapshots || []); setVersions(v?.versions || []);
    }).finally(() => setLoading(false));
  }, [selected, periodDays]);

  if (loading && !report) return <Loading />;
  if (!report) return <div className="text-center text-[#64748B] py-10 text-sm">No competition data yet. AVT needs closed trades first.</div>;

  const gap = report.performance_gap;
  const gapColor = gap > 0.1 ? "#10B981" : gap < -0.1 ? "#EF4444" : "#F59E0B";
  const user = report.user_stats || {};
  const avt = report.avt_stats || {};
  const diffs = report.differences || [];

  const trendData = history.slice().reverse().map((h: any) => ({
    date: h.snapshot_date?.slice(5),
    avt: h.avt_stats?.avg_r ?? 0,
    you: h.user_stats?.avg_r ?? 0,
  }));

  return (
    <div className="space-y-4">
      {/* Selectors */}
      <div className="flex gap-2 items-center">
        {strategies.length > 1 && (
          <select value={selected} onChange={(e) => setSelected(e.target.value)}
            className="bg-[#1A2035] border border-[#2A3050] text-[#F1F5F9] text-xs rounded px-2 py-1">
            {strategies.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        )}
        <select value={periodDays} onChange={(e) => setPeriodDays(Number(e.target.value))}
          className="bg-[#1A2035] border border-[#2A3050] text-[#F1F5F9] text-xs rounded px-2 py-1">
          {[7, 14, 30, 60, 90].map((d) => <option key={d} value={d}>{d} days</option>)}
        </select>
      </div>

      {/* Gap headline */}
      <div className={CARD + " text-center"}>
        <div className="text-[10px] uppercase tracking-widest text-[#64748B] mb-1">Performance Gap (AVT - You)</div>
        <div className="text-3xl font-bold font-num" style={{ color: gapColor }}>
          {gap >= 0 ? "+" : ""}{gap.toFixed(3)}R
        </div>
        <div className="text-xs text-[#64748B] mt-1">per trade over {report.period_days} days</div>
      </div>

      {/* Side-by-side */}
      <div className="grid grid-cols-2 gap-3">
        <div className={CARD}>
          <h3 className="text-xs font-semibold text-[#3B82F6] uppercase tracking-wider mb-3">You</h3>
          <StatsGrid stats={user} color="#3B82F6" />
        </div>
        <div className={CARD}>
          <h3 className="text-xs font-semibold text-[#8B5CF6] uppercase tracking-wider mb-3">AVT</h3>
          <StatsGrid stats={avt} color="#8B5CF6" />
        </div>
      </div>

      {/* Differences */}
      {diffs.length > 0 && (
        <div className={CARD}>
          <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider mb-3">Key Differences</h3>
          {diffs.slice(0, 5).map((d: any, i: number) => (
            <div key={i} className="border-l-2 pl-3 py-1.5 mb-2" style={{ borderColor: d.impact_r > 0 ? "#10B981" : "#EF4444" }}>
              <div className="flex justify-between items-start">
                <span className="text-xs text-[#F1F5F9]">{d.description}</span>
                <span className={`text-xs font-num font-bold ml-2 shrink-0 ${d.impact_r > 0 ? "text-[#10B981]" : "text-[#EF4444]"}`}>
                  {d.impact_r > 0 ? "+" : ""}{d.impact_r?.toFixed(2)}R
                </span>
              </div>
              <span className="text-[10px] text-[#64748B] uppercase">{d.category}</span>
              {d.actionable_insight && <p className="text-[10px] text-[#A78BFA] mt-0.5">{d.actionable_insight}</p>}
            </div>
          ))}
        </div>
      )}

      {/* Trend chart */}
      {trendData.length > 2 && (
        <div className={CARD}>
          <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider mb-3">Avg R Trend</h3>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={trendData}>
              <CartesianGrid stroke="#2A3050" strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fill: "#64748B", fontSize: 10 }} />
              <YAxis tick={{ fill: "#64748B", fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#1A2035", border: "1px solid #2A3050", borderRadius: 8, fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Line type="monotone" dataKey="avt" stroke="#8B5CF6" strokeWidth={2} dot={false} name="AVT" />
              <Line type="monotone" dataKey="you" stroke="#3B82F6" strokeWidth={2} dot={false} name="You" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Narrative */}
      {report.overall_narrative && (
        <div className={CARD}>
          <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider mb-2">AVT's Analysis</h3>
          <p className="text-xs text-[#94A3B8] leading-relaxed whitespace-pre-line">{report.overall_narrative}</p>
        </div>
      )}

      {/* Version timeline */}
      {versions.length > 0 && (
        <div className={CARD}>
          <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider mb-3">Version History</h3>
          {versions.map((v: any, i: number) => (
            <div key={v.id || i} className="flex items-center gap-3 mb-2">
              <div className={`w-7 h-7 rounded-full flex items-center justify-center text-[10px] font-bold ${!v.ended_at ? "bg-[#8B5CF6]/20 text-[#C4B5FD]" : "bg-[#1E2740] text-[#64748B]"}`}>
                {v.version}
              </div>
              <div className="flex-1 text-xs text-[#94A3B8]">
                {v.trade_count} trades
                {v.performance_stats?.expectancy_r != null && ` | ${v.performance_stats.expectancy_r.toFixed(2)}R avg`}
                <span className="text-[10px] text-[#64748B] ml-2">
                  {v.started_at?.slice(0, 10)}{v.ended_at ? ` → ${v.ended_at.slice(0, 10)}` : " (current)"}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════
//  IMPROVEMENTS TAB
// ══════════════════════════════════════════════════════════════════

function ImprovementsTab() {
  const [filter, setFilter] = useState<string>("proven");
  const [improvements, setImprovements] = useState<any[]>([]);
  const [selected, setSelected] = useState<any>(null);
  const [presentation, setPresentation] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [presLoading, setPresLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    avtApi.improvements({ status: filter === "all" ? undefined : filter })
      .then((r) => setImprovements(r.improvements))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [filter]);

  const openDetail = async (imp: any) => {
    setSelected(imp);
    setPresentation(null);
    if (imp.status === "proven") {
      setPresLoading(true);
      try { setPresentation(await avtApi.improvementPresentation(imp.id)); } catch {}
      setPresLoading(false);
    }
  };

  const handleAdopt = async (id: string) => {
    try {
      await avtApi.adoptImprovement(id);
      const r = await avtApi.improvements({ status: filter === "all" ? undefined : filter });
      setImprovements(r.improvements);
      setSelected(null);
    } catch {}
  };

  return (
    <div className="space-y-3">
      {/* Filter */}
      <div className="flex gap-1">
        {["proven", "active", "monitoring", "reverted", "all"].map((f) => (
          <button key={f} onClick={() => { setFilter(f); setSelected(null); }}
            className={`px-3 py-1 text-[10px] uppercase tracking-wider rounded transition-colors ${
              filter === f ? "bg-[#8B5CF6]/20 text-[#C4B5FD]" : "text-[#64748B] hover:text-[#94A3B8]"
            }`}>{f}</button>
        ))}
      </div>

      {loading && <Loading />}

      {!loading && improvements.length === 0 && (
        <div className="text-center text-[#64748B] py-10 text-sm">No {filter === "all" ? "" : filter} improvements yet</div>
      )}

      <div className="flex gap-4">
        {/* List */}
        <div className={`${selected ? "w-2/5" : "w-full"} space-y-2 transition-all`}>
          {improvements.map((imp) => {
            const sc: Record<string, string> = { proven: "#10B981", active: "#3B82F6", monitoring: "#F59E0B", reverted: "#EF4444" };
            const c = sc[imp.status] || "#64748B";
            return (
              <button key={imp.id} onClick={() => openDetail(imp)}
                className={`w-full text-left rounded-lg p-3 border transition-all ${
                  selected?.id === imp.id ? "border-[#8B5CF6] bg-[#8B5CF6]/10" : "border-[#2A3050] bg-[#1A2035] hover:border-[#3A4060]"
                }`}>
                <div className="flex justify-between items-start mb-1">
                  <span className="text-xs text-[#F1F5F9] line-clamp-2">{imp.description}</span>
                  <span className="shrink-0 ml-2 text-[9px] uppercase px-1.5 py-0.5 rounded font-medium" style={{ color: c, backgroundColor: c + "20" }}>{imp.status}</span>
                </div>
                <div className="flex gap-3 text-[10px] text-[#64748B]">
                  <span className="uppercase">{imp.type?.replace(/_/g, " ")}</span>
                  {imp.evidence_strength > 0 && <span>Evidence: <span className="text-[#A78BFA] font-num">{(imp.evidence_strength * 100).toFixed(0)}%</span></span>}
                </div>
              </button>
            );
          })}
        </div>

        {/* Detail panel */}
        {selected && (
          <div className="flex-1 space-y-3 overflow-y-auto">
            <button onClick={() => setSelected(null)} className="text-[10px] text-[#64748B] hover:text-[#94A3B8]">&larr; Close</button>

            <div className={CARD}>
              <h2 className="text-sm font-bold text-[#F1F5F9] mb-1">{selected.description}</h2>
              <div className="flex flex-wrap gap-3 text-[10px] text-[#64748B]">
                <span>Type: <span className="text-[#94A3B8] uppercase">{selected.type?.replace(/_/g, " ")}</span></span>
                <span>Version: <span className="text-[#94A3B8]">v{selected.applied_at_version}</span></span>
                <span>Evidence: <span className="text-[#A78BFA] font-num">{(selected.evidence_strength * 100).toFixed(0)}%</span></span>
              </div>
            </div>

            {/* Params */}
            {selected.params && Object.keys(selected.params).length > 0 && (
              <div className={CARD}>
                <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider mb-2">Parameters</h3>
                <div className="grid grid-cols-2 gap-1">
                  {Object.entries(selected.params).map(([k, v]) => (
                    <div key={k} className="text-xs"><span className="text-[#64748B]">{k.replace(/_/g, " ")}:</span> <span className="text-[#F1F5F9] font-num">{String(v)}</span></div>
                  ))}
                </div>
              </div>
            )}

            {/* Before/After */}
            {(selected.performance_before || selected.performance_after) && (
              <div className="grid grid-cols-2 gap-3">
                {selected.performance_before && (
                  <div className={CARD}><h3 className="text-xs font-semibold text-[#EF4444]/80 uppercase mb-2">Before</h3><PerfBlock perf={selected.performance_before} /></div>
                )}
                {selected.performance_after && (
                  <div className={CARD}><h3 className="text-xs font-semibold text-[#10B981] uppercase mb-2">After</h3><PerfBlock perf={selected.performance_after} /></div>
                )}
              </div>
            )}

            {/* Revert reason */}
            {selected.status === "reverted" && selected.revert_reason && (
              <div className={CARD + " border-[#EF4444]/30"}>
                <h3 className="text-xs font-semibold text-[#EF4444] uppercase mb-1">Reverted</h3>
                <p className="text-xs text-[#94A3B8]">{selected.revert_reason}</p>
              </div>
            )}

            {/* Presentation */}
            {presLoading && <div className="text-center text-[#64748B] py-4 text-xs">Generating presentation...</div>}
            {presentation && (
              <div className={CARD + " border-[#8B5CF6]/30"}>
                <h3 className="text-sm font-bold text-[#C4B5FD] mb-2">{presentation.title}</h3>
                <p className="text-xs text-[#94A3B8] leading-relaxed whitespace-pre-line mb-3">{presentation.narrative}</p>

                {presentation.examples?.length > 0 && (
                  <div className="space-y-1 mb-3">
                    <h4 className="text-[10px] text-[#64748B] uppercase">Examples</h4>
                    {presentation.examples.map((ex: any, i: number) => (
                      <div key={i} className="bg-[#0A0E17] rounded p-2 text-xs flex justify-between">
                        <span className="text-[#F1F5F9]">{ex.symbol}</span>
                        <span className="text-[10px]">
                          With: <span className="text-[#10B981] font-num">{ex.with_improvement_r?.toFixed(1)}R</span>
                          {" "}Without: <span className="text-[#EF4444] font-num">{ex.without_improvement_r?.toFixed(1)}R</span>
                          {" "}Diff: <span className="text-[#8B5CF6] font-num">{ex.difference > 0 ? "+" : ""}{ex.difference?.toFixed(1)}R</span>
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Confidence bar */}
                <div className="mb-3">
                  <div className="flex justify-between text-[10px] text-[#64748B] mb-1">
                    <span>Confidence</span><span className="font-num">{(presentation.avt_confidence * 100).toFixed(0)}%</span>
                  </div>
                  <div className="w-full h-1.5 bg-[#1E2740] rounded-full overflow-hidden">
                    <div className="h-full bg-[#8B5CF6] rounded-full" style={{ width: `${presentation.avt_confidence * 100}%` }} />
                  </div>
                </div>

                {/* Actions */}
                {selected.status === "proven" && (
                  <div className="flex gap-2">
                    <button onClick={() => handleAdopt(selected.id)}
                      className="px-3 py-1.5 text-xs bg-[#10B981]/20 text-[#10B981] rounded hover:bg-[#10B981]/30 font-medium transition-colors">
                      Adopt as Rule
                    </button>
                    <button className="px-3 py-1.5 text-xs bg-[#1E2740] text-[#94A3B8] rounded hover:bg-[#2A3050] transition-colors">
                      Keep Virtual Only
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════
//  SCAN LOGS TAB
// ══════════════════════════════════════════════════════════════════

// ══════════════════════════════════════════════════════════════════
//  CHAT TAB — Talk to AVT
// ══════════════════════════════════════════════════════════════════

function ChatTab() {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  // Load history on mount
  useEffect(() => {
    avtApi.chatHistory(30).then((r) => {
      if (r?.history) setMessages(r.history);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    const msg = input.trim();
    if (!msg || sending) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: msg }]);
    setSending(true);
    try {
      const res = await avtApi.chat(msg);
      setMessages((prev) => [...prev, { role: "avt", content: res.response }]);
    } catch {
      setMessages((prev) => [...prev, { role: "avt", content: "Sorry, I couldn't process that. Try again." }]);
    }
    setSending(false);
  };

  const quickActions = [
    "What are you working on?",
    "How did I do today?",
    "What should I improve?",
    "Show my performance",
  ];

  return (
    <div className="flex flex-col h-full max-h-[calc(100vh-180px)]">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-3 pb-4">
        {messages.length === 0 && (
          <div className="text-center py-10">
            <div className="text-2xl mb-2">🤖</div>
            <p className="text-sm text-[#94A3B8] mb-1">Talk to AVT</p>
            <p className="text-[10px] text-[#64748B] mb-4">
              Ask about your performance, what AVT is testing, or get suggestions
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {quickActions.map((q) => (
                <button key={q} onClick={() => { setInput(q); }}
                  className="px-3 py-1.5 text-xs bg-[#1A2035] border border-[#2A3050] rounded-lg text-[#94A3B8] hover:text-[#F1F5F9] hover:border-[#3A4060] transition-colors">
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-[70%] rounded-lg px-3 py-2 text-xs leading-relaxed whitespace-pre-line ${
              m.role === "user"
                ? "bg-[#8B5CF6]/20 text-[#C4B5FD] border border-[#8B5CF6]/30"
                : "bg-[#1A2035] text-[#94A3B8] border border-[#2A3050]"
            }`}>
              {m.role === "avt" && <span className="text-[9px] text-[#64748B] block mb-1 uppercase">AVT</span>}
              {m.content}
            </div>
          </div>
        ))}
        {sending && (
          <div className="flex justify-start">
            <div className="bg-[#1A2035] border border-[#2A3050] rounded-lg px-3 py-2 text-xs text-[#64748B]">
              Thinking...
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Input */}
      <div className="border-t border-[#2A3050] pt-3 flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Ask AVT anything..."
          className="flex-1 bg-[#1A2035] border border-[#2A3050] rounded-lg px-3 py-2 text-xs text-[#F1F5F9] placeholder:text-[#64748B] focus:outline-none focus:border-[#8B5CF6]/50"
        />
        <button onClick={send} disabled={sending || !input.trim()}
          className="px-4 py-2 bg-[#8B5CF6]/20 text-[#C4B5FD] text-xs font-medium rounded-lg hover:bg-[#8B5CF6]/30 transition-colors disabled:opacity-40">
          Send
        </button>
      </div>
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════
//  ACTIVITY TAB — Observations, scan logs, triage messages
// ══════════════════════════════════════════════════════════════════

function ActivityTab() {
  const [subTab, setSubTab] = useState<"observations" | "scans" | "proving">("observations");
  const [observations, setObservations] = useState<any[]>([]);
  const [logs, setLogs] = useState<any[]>([]);
  const [proving, setProving] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    if (subTab === "observations") {
      avtApi.todayObservations()
        .then((r) => setObservations(r?.observations || []))
        .catch(() => setObservations([]))
        .finally(() => setLoading(false));
    } else if (subTab === "scans") {
      avtApi.scanLogs(50)
        .then((r) => setLogs(r.logs))
        .catch(() => setLogs([]))
        .finally(() => setLoading(false));
    } else {
      avtApi.provingStatus()
        .then((r) => setProving(r || {}))
        .catch(() => setProving({}))
        .finally(() => setLoading(false));
    }
  }, [subTab]);

  const severityColors: Record<string, string> = {
    critical: "#EF4444",
    warning: "#F59E0B",
    info: "#3B82F6",
    positive: "#10B981",
  };

  return (
    <div className="space-y-3">
      {/* Sub-tabs */}
      <div className="flex gap-2">
        {([
          { key: "observations", label: "Observations" },
          { key: "scans", label: "Scan Logs" },
          { key: "proving", label: "AVT Proving" },
        ] as const).map((t) => (
          <button key={t.key} onClick={() => setSubTab(t.key)}
            className={`px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider rounded transition-colors ${
              subTab === t.key ? "bg-[#8B5CF6]/15 text-[#C4B5FD]" : "text-[#64748B] hover:text-[#94A3B8]"
            }`}>{t.label}</button>
        ))}
      </div>

      {loading && <Loading />}

      {/* Observations */}
      {!loading && subTab === "observations" && (
        <div className="space-y-2">
          {observations.length === 0 && (
            <div className="text-center text-[#64748B] py-10 text-sm">No observations today yet</div>
          )}
          {observations.map((o: any) => (
            <div key={o.id} className={CARD + " py-3"}>
              <div className="flex items-start justify-between mb-1">
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: severityColors[o.severity] || "#64748B" }} />
                  <span className="text-xs font-medium text-[#F1F5F9]">{o.title}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[9px] uppercase px-1.5 py-0.5 rounded font-medium"
                    style={{ color: severityColors[o.severity], backgroundColor: (severityColors[o.severity] || "#64748B") + "20" }}>
                    {o.severity}
                  </span>
                </div>
              </div>
              <p className="text-[10px] text-[#94A3B8] leading-relaxed ml-4">{o.description}</p>
              <div className="flex gap-3 text-[9px] text-[#64748B] mt-1.5 ml-4">
                <span>{o.type?.replace(/_/g, " ")}</span>
                <span>{o.timestamp?.slice(11, 19)}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Scan logs */}
      {!loading && subTab === "scans" && (
        <div className="space-y-2">
          {logs.length === 0 && (
            <div className="text-center text-[#64748B] py-10 text-sm">No scan logs yet</div>
          )}
          <p className="text-[10px] text-[#64748B] mb-2">{logs.length} recent scan cycles</p>
          {logs.map((l: any) => (
            <div key={l.id} className={CARD + " py-3 flex items-center justify-between"}>
              <div className="flex gap-4 items-center">
                <span className="text-[10px] text-[#64748B] font-num w-16">{l.cycle_time?.slice(11, 19)}</span>
                <span className="text-xs text-[#94A3B8]">{l.candidates_found} candidates</span>
                {l.trades_opened > 0 && <span className="text-xs text-[#10B981]">+{l.trades_opened} opened</span>}
                {l.trades_closed > 0 && <span className="text-xs text-[#F59E0B]">{l.trades_closed} closed</span>}
              </div>
              <div className="flex gap-3 items-center">
                {l.skipped?.length > 0 && <span className="text-[10px] text-[#64748B]">{l.skipped.length} skipped</span>}
                <span className="text-[10px] text-[#64748B] font-num">{l.duration_ms}ms</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* What AVT is proving with its own trades */}
      {!loading && subTab === "proving" && (
        <div className="space-y-2">
          {Object.keys(proving).length === 0 && (
            <div className="text-center text-[#64748B] py-10 text-sm">
              AVT hasn't started proving any improvements yet.
              <br />
              <span className="text-[10px]">It needs 30+ virtual trades per strategy before it can analyze.</span>
            </div>
          )}
          {Object.entries(proving).map(([sid, s]: [string, any]) => (
            <div key={sid} className={CARD + " py-3"}>
              <div className="flex justify-between items-start mb-2">
                <span className="text-xs font-medium text-[#F1F5F9]">{sid}</span>
                <span className="text-[10px] text-[#64748B]">
                  {s.proven_count > 0 ? `${s.proven_count} proven (+${s.proven_total_impact_r?.toFixed(3)}R)` : "No proven improvements yet"}
                </span>
              </div>
              {s.proving ? (
                <div className="bg-[#0A0E17] rounded p-2">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[9px] uppercase px-1.5 py-0.5 rounded font-medium text-[#F59E0B] bg-[#F59E0B]/20">
                      Proving
                    </span>
                    <span className="text-xs text-[#94A3B8]">{s.proving.title}</span>
                  </div>
                  <div className="flex gap-3 text-[9px] text-[#64748B]">
                    <span>{s.proving.trades}/{s.proving.needed} trades</span>
                  </div>
                  <div className="mt-1.5 h-1 bg-[#1A2035] rounded-full overflow-hidden">
                    <div className="h-full bg-[#F59E0B] rounded-full transition-all"
                      style={{ width: `${Math.min(100, (s.proving.trades / s.proving.needed) * 100)}%` }} />
                  </div>
                </div>
              ) : (
                <p className="text-[10px] text-[#64748B]">No active proof — waiting for next analysis cycle</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════
//  SHARED COMPONENTS
// ══════════════════════════════════════════════════════════════════

// ══════════════════════════════════════════════════════════════════
//  STRATEGY HEALTH TAB
// ══════════════════════════════════════════════════════════════════

const STATE_COLORS: Record<string, string> = {
  autonomous: "#10B981",
  ready: "#10B981",
  probation: "#F59E0B",
  calibrating: "#3B82F6",
  degraded: "#F97316",
  paused: "#EF4444",
  rejected: "#EF4444",
  ingested: "#64748B",
  health_check: "#64748B",
};

const GRADE_COLORS: Record<string, string> = {
  ready: "#10B981",
  probation: "#F59E0B",
  rejected: "#EF4444",
};

function StrategyHealthTab() {
  const [allLifecycle, setAllLifecycle] = useState<Record<string, any>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<any>(null);
  const [calReport, setCalReport] = useState<any>(null);
  const [edgeMon, setEdgeMon] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionMsg, setActionMsg] = useState<string | null>(null);
  const mountedRef = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const res = await avtApi.lifecycleAll();
      if (!mountedRef.current) return;
      setAllLifecycle(res.strategies || {});
      setError(null);
    } catch (e: any) {
      if (mountedRef.current) setError(e.message);
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    refresh();
    return () => { mountedRef.current = false; };
  }, [refresh]);

  const selectStrategy = useCallback(async (sid: string) => {
    setSelected(sid);
    setDetail(null);
    setCalReport(null);
    setEdgeMon(null);
    try {
      const [info, edge] = await Promise.all([
        avtApi.lifecycle(sid),
        avtApi.edgeMonitor(sid).catch(() => null),
      ]);
      if (!mountedRef.current) return;
      setDetail(info);
      setEdgeMon(edge);
      if (info.state === "calibrating") {
        const cal = await avtApi.calibrationReport(sid).catch(() => null);
        if (mountedRef.current) setCalReport(cal);
      }
    } catch (e: any) {
      if (mountedRef.current) setError(e.message);
    }
  }, []);

  const doAction = useCallback(async (action: string, sid: string) => {
    setActionMsg(null);
    try {
      if (action === "start_calibration") await avtApi.startCalibration(sid);
      else if (action === "skip_calibration") await avtApi.skipCalibration(sid);
      else if (action === "graduate") await avtApi.graduateCalibration(sid);
      else if (action === "resume") await avtApi.resumeStrategy(sid);
      setActionMsg("Done");
      refresh();
      selectStrategy(sid);
    } catch (e: any) {
      setActionMsg(e.message);
    }
  }, [refresh, selectStrategy]);

  if (loading) return <Loading />;
  if (error && Object.keys(allLifecycle).length === 0) return <Error msg={error} />;

  const strategies = Object.entries(allLifecycle);

  return (
    <div className="space-y-4">
      {/* Overview Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
        {strategies.map(([sid, info]) => (
          <button key={sid} onClick={() => selectStrategy(sid)}
            className={`${CARD} text-left transition-all hover:border-[#8B5CF6]/50 ${selected === sid ? "border-[#8B5CF6]" : ""}`}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-[#F1F5F9] font-medium truncate max-w-[120px]">{sid}</span>
              <span className="text-[9px] font-bold uppercase px-2 py-0.5 rounded-full"
                style={{ color: STATE_COLORS[info.state] || "#64748B", backgroundColor: (STATE_COLORS[info.state] || "#64748B") + "20" }}>
                {info.state}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <div>
                <div className="text-[10px] text-[#64748B]">Grade</div>
                <div className="text-sm font-bold" style={{ color: GRADE_COLORS[info.grade] || "#64748B" }}>
                  {info.grade?.toUpperCase() || "—"}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-[#64748B]">Health</div>
                <div className="text-sm font-bold text-[#F1F5F9]">{(info.health_score ?? 0).toFixed(0)}</div>
              </div>
              <div>
                <div className="text-[10px] text-[#64748B]">Alloc</div>
                <div className="text-sm font-bold text-[#A78BFA]">{((info.allocation_multiplier ?? 0) * 100).toFixed(0)}%</div>
              </div>
            </div>
            {info.degradation_level && info.degradation_level !== "healthy" && (
              <div className="mt-2 text-[9px] px-2 py-1 rounded bg-[#EF4444]/10 text-[#F87171]">
                Edge: {info.degradation_level}
              </div>
            )}
          </button>
        ))}
        {strategies.length === 0 && (
          <div className="col-span-full text-center text-xs text-[#64748B] py-8">No strategies loaded</div>
        )}
      </div>

      {/* Action feedback */}
      {actionMsg && (
        <div className="text-xs text-[#C4B5FD] bg-[#8B5CF6]/10 border border-[#8B5CF6]/30 rounded px-3 py-2">
          {actionMsg}
        </div>
      )}

      {/* Detail Panel */}
      {selected && detail && (
        <div className={CARD}>
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-bold text-[#F1F5F9]">{selected}</h3>
              <div className="flex gap-2 mt-1">
                <span className="text-[9px] font-bold uppercase px-2 py-0.5 rounded-full"
                  style={{ color: STATE_COLORS[detail.state] || "#64748B", backgroundColor: (STATE_COLORS[detail.state] || "#64748B") + "20" }}>
                  {detail.state}
                </span>
                <span className="text-[9px] font-bold uppercase px-2 py-0.5 rounded-full"
                  style={{ color: GRADE_COLORS[detail.grade] || "#64748B", backgroundColor: (GRADE_COLORS[detail.grade] || "#64748B") + "20" }}>
                  {detail.grade || "—"}
                </span>
              </div>
            </div>
            <div className="flex gap-2">
              {(detail.state === "autonomous" || detail.state === "probation") && (
                <button onClick={() => doAction("start_calibration", selected)}
                  className="text-[10px] bg-[#3B82F6]/20 text-[#60A5FA] px-3 py-1.5 rounded hover:bg-[#3B82F6]/30">
                  Start Calibration
                </button>
              )}
              {detail.state === "calibrating" && (
                <>
                  <button onClick={() => doAction("skip_calibration", selected)}
                    className="text-[10px] bg-[#64748B]/20 text-[#94A3B8] px-3 py-1.5 rounded hover:bg-[#64748B]/30">
                    Skip
                  </button>
                  <button onClick={() => doAction("graduate", selected)}
                    className="text-[10px] bg-[#10B981]/20 text-[#34D399] px-3 py-1.5 rounded hover:bg-[#10B981]/30">
                    Graduate
                  </button>
                </>
              )}
              {(detail.state === "degraded" || detail.state === "paused") && (
                <button onClick={() => doAction("resume", selected)}
                  className="text-[10px] bg-[#F59E0B]/20 text-[#FBBF24] px-3 py-1.5 rounded hover:bg-[#F59E0B]/30">
                  Resume
                </button>
              )}
            </div>
          </div>

          {/* Health Score Bar */}
          <div className="mb-4">
            <div className="flex justify-between text-[10px] text-[#64748B] mb-1">
              <span>Health Score</span>
              <span className="text-[#F1F5F9] font-bold">{(detail.health_score ?? 0).toFixed(0)} / 100</span>
            </div>
            <div className="h-2 bg-[#0F1629] rounded-full overflow-hidden">
              <div className="h-full rounded-full transition-all"
                style={{
                  width: `${Math.min(100, detail.health_score ?? 0)}%`,
                  backgroundColor: (detail.health_score ?? 0) >= 70 ? "#10B981" : (detail.health_score ?? 0) >= 40 ? "#F59E0B" : "#EF4444",
                }} />
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
            <MetricCard label="Allocation" value={`${((detail.allocation_multiplier ?? 0) * 100).toFixed(0)}%`}
              color="#A78BFA" />
            <MetricCard label="Degradation" value={detail.degradation_level || "healthy"}
              color={detail.degradation_level === "healthy" ? "#10B981" : detail.degradation_level === "warning" ? "#F59E0B" : "#EF4444"} />
            <MetricCard label="Rolling Exp"
              value={detail.rolling_expectancy != null ? `${detail.rolling_expectancy.toFixed(3)}R` : "—"}
              color={detail.rolling_expectancy > 0 ? "#10B981" : "#EF4444"} />
            <MetricCard label="Translation Drift"
              value={detail.translation_drift != null ? detail.translation_drift.toFixed(3) : "—"}
              color={detail.translation_drift > 0.3 ? "#EF4444" : "#10B981"} />
          </div>

          {/* Health Issues */}
          {detail.health_issues && detail.health_issues.length > 0 && (
            <div className="mb-4">
              <h4 className="text-[10px] text-[#64748B] uppercase mb-2">Health Issues</h4>
              <div className="space-y-1">
                {detail.health_issues.map((issue: any, i: number) => (
                  <div key={i} className={`text-xs px-3 py-2 rounded ${
                    issue.severity === "critical" ? "bg-[#EF4444]/10 text-[#F87171]" :
                    issue.severity === "warning" ? "bg-[#F59E0B]/10 text-[#FBBF24]" :
                    "bg-[#3B82F6]/10 text-[#60A5FA]"
                  }`}>
                    <span className="font-medium">[{issue.category}]</span> {issue.message}
                    {issue.suggestion && (
                      <span className="block text-[10px] mt-0.5 opacity-75">{issue.suggestion}</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Calibration Section */}
          {detail.state === "calibrating" && calReport && (
            <CalibrationPanel report={calReport} strategyId={selected} onRefresh={() => selectStrategy(selected)} />
          )}

          {/* Edge Monitor Section */}
          {edgeMon && edgeMon.level && edgeMon.status !== "no_monitor" && (
            <EdgeMonitorPanel monitor={edgeMon} />
          )}

          {/* Degraded/Paused Alert */}
          {(detail.state === "degraded" || detail.state === "paused") && (
            <div className={`mt-4 p-3 rounded-lg border ${
              detail.state === "paused"
                ? "bg-[#EF4444]/10 border-[#EF4444]/30 text-[#F87171]"
                : "bg-[#F97316]/10 border-[#F97316]/30 text-[#FB923C]"
            }`}>
              <div className="text-xs font-bold uppercase mb-1">
                {detail.state === "paused" ? "Strategy Paused" : "Strategy Degraded"}
              </div>
              <div className="text-[10px] opacity-80">
                {detail.state === "paused"
                  ? "Critical edge loss detected. No new trades. Review and resume when ready."
                  : "Edge degradation detected. Allocation reduced to 50%. Monitor or resume to full."}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}


function CalibrationPanel({ report, strategyId, onRefresh }: { report: any; strategyId: string; onRefresh: () => void }) {
  const [annotateId, setAnnotateId] = useState<string | null>(null);
  const [annotateText, setAnnotateText] = useState("");

  const submitAnnotation = async () => {
    if (!annotateId || !annotateText.trim()) return;
    try {
      await avtApi.annotateCalibration(strategyId, annotateId, annotateText.trim());
      setAnnotateId(null);
      setAnnotateText("");
      onRefresh();
    } catch {}
  };

  const pct = Math.round((report.alignment_score ?? 0) * 100);
  const radius = 36;
  const circ = 2 * Math.PI * radius;
  const offset = circ - (pct / 100) * circ;

  return (
    <div className="mt-4 pt-4 border-t border-[#2A3050]">
      <h4 className="text-[10px] text-[#64748B] uppercase mb-3">Calibration Progress</h4>

      <div className="flex gap-6 items-center mb-4">
        {/* Alignment Ring */}
        <div className="relative w-20 h-20">
          <svg viewBox="0 0 80 80" className="w-full h-full">
            <circle cx="40" cy="40" r={radius} fill="none" stroke="#1E293B" strokeWidth="6" />
            <circle cx="40" cy="40" r={radius} fill="none"
              stroke={pct >= 70 ? "#10B981" : pct >= 50 ? "#F59E0B" : "#EF4444"}
              strokeWidth="6" strokeLinecap="round"
              strokeDasharray={circ} strokeDashoffset={offset}
              transform="rotate(-90 40 40)" />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-sm font-bold text-[#F1F5F9]">{pct}%</span>
            <span className="text-[8px] text-[#64748B]">alignment</span>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-3">
          <MetricCard label="TP" value={report.true_positives} color="#10B981" />
          <MetricCard label="FP" value={report.false_positives} color="#EF4444" />
          <MetricCard label="FN" value={report.false_negatives} color="#F59E0B" />
        </div>

        <div className="text-xs space-y-1">
          <div className="text-[#64748B]">Comparisons: <span className="text-[#F1F5F9]">{report.total_comparisons}</span></div>
          <div className="text-[#64748B]">Signals: <span className="text-[#F1F5F9]">{report.signals_generated}</span></div>
          <div className={report.ready_to_graduate ? "text-[#10B981]" : "text-[#F59E0B]"}>
            {report.ready_to_graduate ? "Ready to graduate" : report.graduation_message}
          </div>
        </div>
      </div>

      {/* Graduation Blockers */}
      {report.graduation_blockers?.length > 0 && (
        <div className="mb-3 space-y-1">
          {report.graduation_blockers.map((b: string, i: number) => (
            <div key={i} className="text-[10px] text-[#FBBF24] bg-[#F59E0B]/10 px-2 py-1 rounded">{b}</div>
          ))}
        </div>
      )}

      {/* Mismatches */}
      {report.mismatches?.length > 0 && (
        <div>
          <h5 className="text-[10px] text-[#64748B] uppercase mb-2">Recent Mismatches</h5>
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {report.mismatches.map((m: any) => (
              <div key={m.id} className="flex items-center justify-between text-xs bg-[#0F1629] rounded px-3 py-2">
                <div className="flex gap-2 items-center">
                  <span className={`text-[9px] font-bold uppercase px-1.5 py-0.5 rounded ${
                    m.type === "false_positive" ? "bg-[#EF4444]/20 text-[#F87171]" : "bg-[#F59E0B]/20 text-[#FBBF24]"
                  }`}>{m.type === "false_positive" ? "FP" : "FN"}</span>
                  <span className="text-[#F1F5F9]">{m.symbol}</span>
                </div>
                {m.user_annotation ? (
                  <span className="text-[10px] text-[#64748B] italic max-w-[200px] truncate">{m.user_annotation}</span>
                ) : (
                  annotateId === m.id ? (
                    <div className="flex gap-1">
                      <input value={annotateText} onChange={(e) => setAnnotateText(e.target.value)}
                        className="bg-[#1A2035] border border-[#2A3050] text-[#F1F5F9] text-[10px] rounded px-2 py-1 w-40"
                        placeholder="Why this mismatch?" onKeyDown={(e) => e.key === "Enter" && submitAnnotation()} />
                      <button onClick={submitAnnotation}
                        className="text-[9px] text-[#10B981] hover:text-[#34D399]">Save</button>
                    </div>
                  ) : (
                    <button onClick={() => setAnnotateId(m.id)}
                      className="text-[9px] text-[#8B5CF6] hover:text-[#A78BFA]">Annotate</button>
                  )
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


function EdgeMonitorPanel({ monitor }: { monitor: any }) {
  const levelColor = monitor.level === "healthy" ? "#10B981"
    : monitor.level === "warning" ? "#F59E0B"
    : monitor.level === "degraded" ? "#F97316" : "#EF4444";

  return (
    <div className="mt-4 pt-4 border-t border-[#2A3050]">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-[10px] text-[#64748B] uppercase">Edge Monitor</h4>
        <span className="text-[9px] font-bold uppercase px-2 py-0.5 rounded-full"
          style={{ color: levelColor, backgroundColor: levelColor + "20" }}>
          {monitor.level}
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="Rolling Exp"
          value={monitor.rolling_expectancy != null ? `${monitor.rolling_expectancy.toFixed(3)}R` : "—"}
          color={monitor.rolling_expectancy > 0 ? "#10B981" : "#EF4444"} />
        <MetricCard label="Rolling WR"
          value={monitor.rolling_win_rate != null ? `${(monitor.rolling_win_rate * 100).toFixed(0)}%` : "—"}
          color="#3B82F6" />
        <MetricCard label="Drift Score"
          value={monitor.drift_score != null ? monitor.drift_score.toFixed(3) : "—"}
          color={monitor.drift_score > 0.3 ? "#EF4444" : "#10B981"} />
        <MetricCard label="Total Trades"
          value={monitor.total_trades ?? 0} color="#A78BFA" />
      </div>

      {monitor.baseline_expectancy != null && (
        <div className="mt-2 text-[10px] text-[#64748B]">
          Baseline: {monitor.baseline_expectancy.toFixed(3)}R exp, {((monitor.baseline_win_rate ?? 0) * 100).toFixed(0)}% WR
        </div>
      )}

      {monitor.message && monitor.message !== "healthy" && (
        <div className="mt-2 text-[10px] px-2 py-1 rounded" style={{ color: levelColor, backgroundColor: levelColor + "10" }}>
          {monitor.message}
        </div>
      )}
    </div>
  );
}


function MetricCard({ label, value, color }: { label: string; value: any; color: string }) {
  return (
    <div className="bg-[#0F1629] rounded px-3 py-2">
      <div className="text-[9px] text-[#64748B] uppercase">{label}</div>
      <div className="text-sm font-bold font-num" style={{ color }}>{value}</div>
    </div>
  );
}


function Loading() {
  return <div className="flex items-center justify-center py-10 text-[#64748B] text-sm">Loading...</div>;
}

function Error({ msg }: { msg: string }) {
  return <div className="flex items-center justify-center py-10 text-[#EF4444] text-sm">{msg}</div>;
}

function StatsGrid({ stats, color }: { stats: any; color: string }) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {[
        { label: "Trades", value: stats.trade_count ?? 0 },
        { label: "Win Rate", value: `${((stats.win_rate ?? 0) * 100).toFixed(0)}%` },
        { label: "Avg R", value: `${(stats.avg_r ?? 0).toFixed(2)}R` },
        { label: "Total R", value: `${(stats.total_r ?? 0).toFixed(1)}R` },
      ].map((i) => (
        <div key={i.label}>
          <div className="text-[10px] text-[#64748B] uppercase">{i.label}</div>
          <div className="text-lg font-bold font-num" style={{ color }}>{i.value}</div>
        </div>
      ))}
    </div>
  );
}

function PerfBlock({ perf }: { perf: Record<string, any> }) {
  return (
    <div className="space-y-1">
      {[
        { label: "Win Rate", value: perf.win_rate != null ? `${(perf.win_rate * 100).toFixed(0)}%` : "—" },
        { label: "Expectancy", value: perf.expectancy_r != null ? `${perf.expectancy_r.toFixed(3)}R` : "—" },
        { label: "Trades", value: perf.trade_count ?? "—" },
        { label: "PF", value: perf.profit_factor != null ? perf.profit_factor.toFixed(2) : "—" },
      ].map((i) => (
        <div key={i.label} className="flex justify-between text-xs">
          <span className="text-[#64748B]">{i.label}</span>
          <span className="text-[#F1F5F9] font-num">{i.value}</span>
        </div>
      ))}
    </div>
  );
}
