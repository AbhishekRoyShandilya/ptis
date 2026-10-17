// src/pages/AnalysisPanel.tsx
// The ANALYSIS (money) thread of a strategy — separate from Teaching. Here you run a real P&L backtest
// (avt_trader: structural stop + asymmetric exit, honest R — not a capped target) and ask money/hypothesis
// questions, answered grounded in the result. Recognition (the eye) is taught elsewhere; this is the money.
import { useState, useRef, useEffect, useCallback } from "react";
import { intelligenceApi } from "@/services/api";

interface Ledger {
  n: number; wins: number; win_rate: number; total_R: number; avg_R: number; avg_win_R: number;
  avg_loss_R: number; expectancy_R: number; profit_factor: number | null; max_dd_R: number;
  best_R: number; worst_R: number; sessions_run?: number; sessions_total?: number; ran_at?: string; partial?: boolean;
}
interface Status { running: boolean; total: number; done: number; result: Ledger | null; eta_sec?: number | null }
interface Msg { id: number; role: "user" | "bot"; text: string }
interface Strat { name: string; direction: string; require: string[]; exclude: string[] }
interface Discovery { kind: string; dimension: string; bucket?: string; headline: string; detail: string; confidence: string; evidence_n: number; refinement?: { op: string; tags: string[]; label: string; proj_exp_delta?: number; n_after?: number } }
interface DiscoverResp { ready: boolean; n: number; total_R?: number; discoveries: Discovery[] }
interface Hyp { text: string; result: string; status?: string; ts: string }

let _aid = 0;
const r2 = (x?: number | null) => (x === undefined || x === null ? "—" : Math.round(x * 100) / 100);
const fmtEta = (s: number) => (s < 60 ? `${Math.round(s)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`);

export default function AnalysisPanel({ strat }: { strat: Strat }) {
  const [status, setStatus] = useState<Status>({ running: false, total: 0, done: 0, result: null });
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [disc, setDisc] = useState<DiscoverResp | null>(null);
  const [hist, setHist] = useState<Hyp[]>([]);
  // local signature: an applied refinement updates this (decoupled from the stale sidebar prop)
  const [sig, setSig] = useState({ direction: strat.direction, require: strat.require, exclude: strat.exclude });
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const awaitingVerdictRef = useRef(false);
  const endRef = useRef<HTMLDivElement>(null);
  const push = (m: Omit<Msg, "id">) => setMsgs((p) => [...p, { ...m, id: _aid++ }]);

  const fetchDisc = useCallback(async (name: string) => {
    try { setDisc(await intelligenceApi.analysisDiscoveries(name) as DiscoverResp); } catch { /* */ }
    try { setHist(((await intelligenceApi.analysisHistory(name) as { hypotheses: Hyp[] }).hypotheses) || []); } catch { /* */ }
  }, []);

  const poll = useCallback(async (name: string) => {
    try {
      const s = await intelligenceApi.analysisStatus(name) as Status;
      setStatus(s);
      if (!s.running && pollRef.current) {
        clearInterval(pollRef.current); pollRef.current = null;
        fetchDisc(name);
        if (awaitingVerdictRef.current) {
          awaitingVerdictRef.current = false;
          try {
            const v = await intelligenceApi.analysisVerdict(name) as { held?: boolean; label?: string; delta_exp?: number; thin?: boolean; baseline?: { total_R: number; expectancy_R: number; n: number }; new?: { total_R: number; expectancy_R: number; n: number } };
            if (v && v.held !== undefined && v.baseline && v.new) {
              setMsgs((p) => [...p, { id: _aid++, role: "bot", text: `Verdict — ${v.label}: ${v.held ? "✓ held" : "✗ didn't hold"}. Expectancy ${v.baseline!.expectancy_R}→${v.new!.expectancy_R}R/trade (${(v.delta_exp ?? 0) >= 0 ? "+" : ""}${v.delta_exp}), total ${v.baseline!.total_R}→${v.new!.total_R}R over ${v.baseline!.n}→${v.new!.n} trades${v.thin ? " · thin sample, treat as provisional" : ""}.${v.held ? "" : " Consider reverting."}` }]);
            }
          } catch { /* */ }
        }
      }
    } catch { /* keep polling */ }
  }, [fetchDisc]);

  // on strategy change: load last ledger + discoveries + the analysis conversation
  useEffect(() => {
    setMsgs([{ id: _aid++, role: "bot", text: `Analysis of “${strat.name}” — the money side. Run a backtest to measure real R (structural stop + asymmetric exit, honest — not a capped target); I'll surface what I find, then you can ask me anything.` }]);
    setDisc(null); setHist([]);
    setSig({ direction: strat.direction, require: strat.require, exclude: strat.exclude });
    (async () => {
      try {
        const s = await intelligenceApi.analysisStatus(strat.name) as Status;
        setStatus(s);
        if (s.running && !pollRef.current) pollRef.current = setInterval(() => poll(strat.name), 4000);
        else fetchDisc(strat.name);
        const sess = await intelligenceApi.teachSession(strat.name) as { analysis_messages?: Msg[] };
        const restored = (sess.analysis_messages || []).map((m) => ({ ...m, id: _aid++ }));
        if (restored.length) setMsgs((p) => [...p, ...restored]);
      } catch { /* */ }
    })();
    return () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; } };
  }, [strat.name, poll, fetchDisc]);   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs, status]);

  async function applyRefinement(op: string, tags: string[], reason: string, label: string) {
    if (status.running || busy) return;
    setBusy(true);
    try {
      const r = await intelligenceApi.analysisApplyRefinement({ name: strat.name, op, tags, reason }) as { require: string[]; exclude: string[]; direction: string };
      setSig({ direction: r.direction, require: r.require, exclude: r.exclude });
      awaitingVerdictRef.current = true;
      push({ role: "bot", text: `Applied: ${label} — a new version (provisional). Re-running the backtest; I'll tell you straight whether it held…` });
      const s = await intelligenceApi.analysisBacktest({ name: strat.name, direction: r.direction, require: r.require, exclude: r.exclude, max_sessions: 60 }) as Status;
      setStatus(s); setDisc(null);
      if (!pollRef.current) pollRef.current = setInterval(() => poll(strat.name), 4000);
    } catch (e) { push({ role: "bot", text: "Couldn't apply: " + (e instanceof Error ? e.message : "error") }); }
    finally { setBusy(false); }
  }

  async function runBacktest() {
    if (status.running || busy) return;
    setBusy(true);
    try {
      const s = await intelligenceApi.analysisBacktest({ name: strat.name, direction: sig.direction, require: sig.require, exclude: sig.exclude, max_sessions: 60 }) as Status;
      setStatus(s);
      push({ role: "bot", text: `Backtest started over ${s.total} sessions. A first number shows in a few seconds and builds as it runs — you can leave this tab and come back, it keeps running and saves.` });
      if (!pollRef.current) pollRef.current = setInterval(() => poll(strat.name), 4000);
    } catch (e) { push({ role: "bot", text: "Couldn't start the backtest: " + (e instanceof Error ? e.message : "error") }); }
    finally { setBusy(false); }
  }

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput(""); push({ role: "user", text }); setBusy(true);
    try {
      const a = await intelligenceApi.analysisAsk({ question: text, name: strat.name, direction: sig.direction, require: sig.require, exclude: sig.exclude }) as { answer: string };
      const next = [...msgs, { id: _aid++, role: "user" as const, text }, { id: _aid++, role: "bot" as const, text: a.answer }];
      push({ role: "bot", text: a.answer });
      intelligenceApi.teachSaveSession({ name: strat.name, direction: strat.direction, require: strat.require, exclude: strat.exclude, messages: next, thread: "analysis" }).catch(() => {});
    } catch (e) { push({ role: "bot", text: "Something went wrong: " + (e instanceof Error ? e.message : "error") }); }
    finally { setBusy(false); }
  }

  const led = status.result;
  const pos = led ? led.total_R >= 0 : true;
  return (
    <div className="flex-1 flex flex-col min-w-0">
      <div className="flex-1 overflow-y-auto px-1 py-3 space-y-3">
        {/* DISCOVERIES FIRST — grounded findings (every one computed from real trades, with evidence) */}
        {disc?.ready && disc.discoveries.length > 0 && (
          <div className="px-3.5 py-3 rounded-2xl bg-[#1A2035] border border-[#2A3050]">
            <div className="text-sm font-semibold text-[#E2E8F0] mb-0.5">I analyzed “{strat.name}” — I found {disc.discoveries.length} thing{disc.discoveries.length > 1 ? "s" : ""}.</div>
            <div className="text-[10px] text-[#64748B] mb-2.5">each is computed from your {disc.n} real trades — click to see the evidence, Apply to test it as a refinement</div>
            <div className="space-y-2">{disc.discoveries.map((d, i) => <Observation key={i} d={d} onApply={applyRefinement} busy={busy || status.running} />)}</div>
          </div>
        )}
        {/* the numbers, now as EVIDENCE underneath */}
        <div className="px-3.5 py-3 rounded-2xl bg-[#1A2035] border border-[#2A3050]">
          <div className="flex items-center justify-between mb-2">
            <div className="text-xs font-semibold text-[#A78BFA]">P&L backtest <span className="text-[#64748B] font-normal">· disciplined simulated R, not brokerage P&L</span></div>
            <button onClick={runBacktest} disabled={status.running || busy} className="px-3 py-1.5 rounded-lg text-xs font-medium bg-[#8B5CF6] text-white hover:bg-[#7C3AED] disabled:opacity-40">{status.running ? "Running…" : led ? "Re-run" : "Run backtest"}</button>
          </div>
          {status.running && (
            <div className="space-y-1 mb-2">
              <div className="h-1.5 rounded-full bg-[#11162A] overflow-hidden"><div className="h-full bg-[#8B5CF6] transition-all" style={{ width: `${status.total ? Math.round(100 * status.done / status.total) : 0}%` }} /></div>
              <div className="text-[11px] text-[#64748B]">replaying {status.done} / {status.total} sessions{status.eta_sec ? ` · ~${fmtEta(status.eta_sec)} left` : ""} — you can leave this tab, it keeps running &amp; saves</div>
            </div>
          )}
          {led && led.n > 0 && (
            <div>
              <div className="flex items-baseline gap-3 mb-2">
                <span className="text-3xl font-bold" style={{ color: pos ? "#10B981" : "#EF4444" }}>{pos ? "+" : ""}{r2(led.total_R)}R</span>
                <span className="text-xs text-[#94A3B8]">over {led.n} trades · {led.sessions_run} sessions</span>
                {(status.running || led.partial) && <span className="px-1.5 py-0.5 rounded text-[10px] bg-[#8B5CF6]/15 text-[#A78BFA] animate-pulse">● live · provisional</span>}
              </div>
              <div className="grid grid-cols-3 gap-2 text-[11px]">
                <Stat k="win rate" v={`${Math.round(led.win_rate * 100)}%`} />
                <Stat k="expectancy" v={`${r2(led.expectancy_R)}R / trade`} />
                <Stat k="profit factor" v={led.profit_factor === null ? "∞" : `${r2(led.profit_factor)}`} />
                <Stat k="avg win" v={`+${r2(led.avg_win_R)}R`} good />
                <Stat k="avg loss" v={`${r2(led.avg_loss_R)}R`} bad />
                <Stat k="max drawdown" v={`${r2(led.max_dd_R)}R`} bad />
              </div>
              <div className="text-[10px] text-[#64748B] mt-2">1R = risk to the structural stop · winners held while the move is alive, exited on story-change/close (no fixed target). {led.ran_at && !led.partial && `Run ${led.ran_at}.`}</div>
            </div>
          )}
          {!status.running && led && led.n === 0 && <div className="text-[12px] text-[#F59E0B]">Backtest ran over {led.sessions_run} sessions but found 0 qualifying trades in that sample — entries didn't confirm. Try Re-run (it samples a wider set).</div>}
          {!status.running && !led && <div className="text-[12px] text-[#64748B]">No backtest yet. Run one to measure the money result.</div>}
        </div>

        {/* improvement history — the strategy's track record of applied refinements + verdicts */}
        {hist.length > 0 && (
          <div className="px-3.5 py-3 rounded-2xl bg-[#1A2035] border border-[#2A3050]">
            <div className="text-xs font-semibold text-[#A78BFA] mb-2">Improvement history</div>
            <div className="space-y-1.5">
              {hist.slice().reverse().map((h, i) => (
                <div key={i} className="text-[11px] border-l-2 pl-2" style={{ borderColor: h.status === "held" ? "#10B981" : h.status === "didnt_hold" ? "#EF4444" : "#2A3050" }}>
                  <div className="text-[#E2E8F0]">{h.text}</div>
                  <div className="text-[10px]" style={{ color: h.status === "held" ? "#10B981" : h.status === "didnt_hold" ? "#EF4444" : "#64748B" }}>{h.result}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {msgs.map((m) => (
          <div key={m.id} className={m.role === "user" ? "flex justify-end" : "flex justify-start"}>
            <div className={`max-w-[85%] px-3 py-2 text-sm whitespace-pre-wrap rounded-2xl ${m.role === "user" ? "rounded-tr-sm bg-[#8B5CF6]/20 border border-[#8B5CF6]/30 text-[#F1F5F9]" : "rounded-tl-sm bg-[#11162A] border border-[#2A3050] text-[#CBD5E1]"}`}>{m.text}</div>
          </div>
        ))}
        {busy && <div className="text-xs text-[#64748B] px-2">thinking…</div>}
        <div ref={endRef} />
      </div>
      <div className="flex items-end gap-2 border-t border-[#2A3050] pt-3 px-1">
        <textarea value={input} onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
          rows={1} placeholder="Ask about the P&L, expectancy, drawdown, or a hypothesis…"
          className="flex-1 bg-[#1A2035] border border-[#2A3050] rounded-xl px-3 py-2.5 text-sm text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6]/50 resize-none" />
        <button onClick={send} disabled={busy || !input.trim()} className="px-4 py-2 rounded-lg text-sm font-medium bg-[#8B5CF6] text-white hover:bg-[#7C3AED] disabled:opacity-40">Send</button>
      </div>
    </div>
  );
}

function Stat({ k, v, good, bad }: { k: string; v: string; good?: boolean; bad?: boolean }) {
  return (
    <div className="px-2 py-1.5 rounded-lg bg-[#11162A] border border-[#2A3050]">
      <div className="text-[10px] text-[#64748B]">{k}</div>
      <div className="font-mono" style={{ color: good ? "#10B981" : bad ? "#EF4444" : "#E2E8F0" }}>{v}</div>
    </div>
  );
}

function Observation({ d, onApply, busy }: { d: Discovery; onApply: (op: string, tags: string[], reason: string, label: string) => void; busy: boolean }) {
  const [open, setOpen] = useState(false);
  const color = d.kind === "loser_cluster" ? "#EF4444" : d.kind === "coverage" ? "#F59E0B"
    : d.confidence === "High" ? "#10B981" : d.confidence === "Medium" ? "#F59E0B" : "#3B82F6";
  const ref = d.refinement;
  return (
    <div onClick={() => setOpen((o) => !o)} className="w-full cursor-pointer px-3 py-2 rounded-lg bg-[#11162A] border border-[#2A3050] hover:border-[#3A4060] transition-colors">
      <div className="flex items-start gap-2.5">
        <span className="mt-1.5 w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
        <div className="flex-1 min-w-0">
          <div className="text-[13px] text-[#E2E8F0] leading-snug">{d.headline}</div>
          <div className="text-[10px] text-[#64748B] mt-0.5">Confidence: {d.confidence} · {d.evidence_n} trades{open ? "" : " · click for evidence"}</div>
          {open && <div className="text-[11px] text-[#94A3B8] mt-1.5 font-mono">{d.detail}</div>}
          {ref && ref.tags.length > 0 && (
            <button onClick={(e) => { e.stopPropagation(); onApply(ref.op, ref.tags, d.headline, ref.label); }} disabled={busy}
              className="mt-2 px-2 py-1 rounded text-[10px] font-medium bg-[#10B981]/15 border border-[#10B981]/40 text-[#10B981] hover:bg-[#10B981]/25 disabled:opacity-40">
              Apply: {ref.label}{ref.proj_exp_delta !== undefined ? ` (proj ${ref.proj_exp_delta >= 0 ? "+" : ""}${ref.proj_exp_delta}R/trade)` : ""} →
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
