// src/components/discovery/ContinuousResearchPanel.tsx
// Continuous Research Mode — user-initiated background research that
// enumerates candidates, runs discovery, and surfaces validated drafts.
//
// User clicks "Send to AVT" on each draft individually to graduate it to a
// strategy YAML file. AVT picks it up via hot-reload and starts virtual-trading.
// NO automatic trading — drafts are inert until user explicitly promotes them.

import { useState, useEffect, useCallback, useRef } from "react";
import {
  researchApi,
  type ResearchTask,
  type DiscoveryDraft,
  type DiscoveryDraftDetail,
} from "@/services/api";

const CARD = "bg-[#131825] border border-[#1E2740] rounded-xl p-4";
const BADGE = "inline-block text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full";

const wrColor = (wr: number) =>
  wr >= 0.65 ? "#10B981" : wr >= 0.55 ? "#F59E0B" : wr >= 0.45 ? "#F97316" : "#EF4444";
const rColor = (r: number) =>
  r >= 2.0 ? "#10B981" : r >= 1.0 ? "#F59E0B" : r >= 0 ? "#F97316" : "#EF4444";

const PATTERN_LIBRARY = [
  "compression_breakout",
  "ema_pullback",
  "vwap_bounce",
  "vwap_reversal",
  "orb_breakout",
  "momentum_exhaustion",
  "gap_fill",
  "pullback_trend",
];

// ── Status badge ────────────────────────────────────────────────
function TaskStatusBadge({ status }: { status: ResearchTask["status"] }) {
  const colors: Record<string, string> = {
    pending: "#64748B",
    running: "#8B5CF6",
    stopped: "#F59E0B",
    completed: "#10B981",
    failed: "#EF4444",
  };
  const c = colors[status] || "#64748B";
  return (
    <span className={BADGE} style={{ color: c, background: `${c}22`, border: `1px solid ${c}55` }}>
      {status}
    </span>
  );
}

function DraftStatusBadge({ status }: { status: DiscoveryDraft["status"] }) {
  const colors: Record<string, string> = {
    pending_review: "#F59E0B",
    sent_to_avt: "#10B981",
    discarded: "#64748B",
  };
  const c = colors[status] || "#64748B";
  return (
    <span className={BADGE} style={{ color: c, background: `${c}22`, border: `1px solid ${c}55` }}>
      {status.replace("_", " ")}
    </span>
  );
}

// ── Draft card ──────────────────────────────────────────────────
function DraftCard({
  draft,
  onSendToAvt,
  onDiscard,
  onView,
  busy,
}: {
  draft: DiscoveryDraft;
  onSendToAvt: (id: string) => void;
  onDiscard: (id: string) => void;
  onView: (id: string) => void;
  busy: boolean;
}) {
  return (
    <div className={CARD + " mb-3"}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h4 className="text-sm font-semibold text-[#F1F5F9] truncate">
              {draft.symbol} <span className="text-[#94A3B8]">·</span> {draft.pattern_type.replace(/_/g, " ")}
            </h4>
            <DraftStatusBadge status={draft.status} />
          </div>
          <div className="text-[11px] text-[#64748B] mt-0.5 truncate">
            {draft.strategy_name}
          </div>
        </div>
        {draft.status === "pending_review" && (
          <div className="flex gap-2 flex-shrink-0">
            <button
              onClick={() => onView(draft.id)}
              disabled={busy}
              className="text-xs px-2 py-1 rounded border border-[#2A3050] text-[#94A3B8] hover:border-[#475569]"
            >
              View
            </button>
            <button
              onClick={() => onDiscard(draft.id)}
              disabled={busy}
              className="text-xs px-2 py-1 rounded border border-[#2A3050] text-[#EF4444] hover:bg-[#EF4444]/10"
            >
              Discard
            </button>
            <button
              onClick={() => onSendToAvt(draft.id)}
              disabled={busy}
              className="text-xs px-3 py-1 rounded bg-[#10B981] text-white hover:bg-[#059669] font-semibold"
            >
              Send to AVT
            </button>
          </div>
        )}
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3 text-xs">
        <div>
          <div className="text-[#64748B]">Sample</div>
          <div className="text-[#F1F5F9] font-mono">{draft.sample_size}</div>
        </div>
        <div>
          <div className="text-[#64748B]">Win Rate</div>
          <div className="font-mono" style={{ color: wrColor(draft.win_rate) }}>
            {(draft.win_rate * 100).toFixed(0)}%
          </div>
        </div>
        <div>
          <div className="text-[#64748B]">Avg R</div>
          <div className="font-mono" style={{ color: rColor(draft.avg_r) }}>
            {draft.avg_r.toFixed(2)}
          </div>
        </div>
        <div>
          <div className="text-[#64748B]">PF</div>
          <div className="text-[#F1F5F9] font-mono">
            {draft.profit_factor != null ? draft.profit_factor.toFixed(2) : "—"}
          </div>
        </div>
      </div>

      {/* Context row (Phase 1 + Phase 2 enrichment) */}
      {(draft.best_regime || draft.htf_alignment || draft.optimal_target_r != null || draft.reversal_cliff_r != null) && (
        <div className="mt-3 pt-3 border-t border-[#1E2740] text-[11px] text-[#94A3B8] space-y-1">
          {draft.best_regime && (
            <div>
              <span className="text-[#64748B]">Best regime: </span>
              <span className="text-[#C4B5FD]">{draft.best_regime}</span>
              {draft.best_time_slot && (
                <span className="text-[#64748B]"> · {draft.best_time_slot}</span>
              )}
            </div>
          )}
          {draft.htf_alignment && (
            <div>
              <span className="text-[#64748B]">HTF: </span>
              <span className="text-[#C4B5FD]">{draft.htf_alignment}</span>
            </div>
          )}
          {draft.optimal_target_r != null && (
            <div>
              <span className="text-[#64748B]">Optimal target: </span>
              <span className="text-[#10B981]">{draft.optimal_target_r.toFixed(1)}R</span>
              {draft.reversal_cliff_r != null && (
                <span className="ml-2">
                  <span className="text-[#64748B]">Reversal cliff: </span>
                  <span className="text-[#EF4444]">{draft.reversal_cliff_r.toFixed(1)}R</span>
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {draft.status === "sent_to_avt" && draft.yaml_path && (
        <div className="mt-3 pt-3 border-t border-[#1E2740] text-[11px] text-[#10B981]">
          ✓ AVT virtual-trading this strategy. YAML: <span className="font-mono">{draft.yaml_path}</span>
        </div>
      )}
    </div>
  );
}

// ── Draft detail modal (simple inline view) ─────────────────────
function DraftDetailView({
  detail,
  onClose,
}: {
  detail: DiscoveryDraftDetail;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-[#0F141F] border border-[#2A3050] rounded-xl p-5 max-w-3xl w-full max-h-[80vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between mb-3">
          <div>
            <h3 className="text-base font-semibold text-[#F1F5F9]">
              {detail.symbol} · {detail.pattern_type.replace(/_/g, " ")}
            </h3>
            <p className="text-xs text-[#64748B] mt-0.5">{detail.strategy_name}</p>
          </div>
          <button
            onClick={onClose}
            className="text-[#94A3B8] hover:text-[#F1F5F9] text-lg leading-none"
          >
            ×
          </button>
        </div>
        <div className="text-[10px] text-[#64748B] uppercase tracking-wider mb-1">
          Strategy YAML
        </div>
        <pre className="bg-[#0A0E1A] border border-[#1E2740] rounded p-3 text-[11px] text-[#C4B5FD] overflow-x-auto whitespace-pre-wrap max-h-96">
{detail.strategy_yaml || "(no YAML)"}
        </pre>
      </div>
    </div>
  );
}

// ── Main panel ──────────────────────────────────────────────────
export default function ContinuousResearchPanel() {
  // Form state
  const [scope, setScope] = useState("nifty50");
  const [theme, setTheme] = useState("");
  const [lookback, setLookback] = useState(2);
  const [similarity, setSimilarity] = useState(85);
  const [maxCandidates, setMaxCandidates] = useState(50);
  const [minSample, setMinSample] = useState(30);
  const [minWinRate, setMinWinRate] = useState(55);
  const [minAvgR, setMinAvgR] = useState(0.5);
  const [selectedPatterns, setSelectedPatterns] = useState<string[]>([...PATTERN_LIBRARY]);

  // Server state
  const [tasks, setTasks] = useState<ResearchTask[]>([]);
  const [drafts, setDrafts] = useState<DiscoveryDraft[]>([]);
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [viewing, setViewing] = useState<DiscoveryDraftDetail | null>(null);
  const pollRef = useRef<number | null>(null);

  // ── Polling for active task / drafts ──
  const refresh = useCallback(async () => {
    try {
      const [taskList, draftList] = await Promise.all([
        researchApi.listTasks(),
        researchApi.listDrafts({ limit: 100 }),
      ]);
      setTasks(taskList);
      setDrafts(draftList);
      // Auto-pick latest running task as the focus
      const running = taskList.find((t) => t.status === "running");
      if (running) setActiveTaskId(running.id);
    } catch (e: any) {
      setError(e.message || "Failed to load research state");
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Poll while a task is running
  useEffect(() => {
    const anyRunning = tasks.some((t) => t.status === "running");
    if (anyRunning) {
      pollRef.current = window.setInterval(refresh, 5000);
      return () => {
        if (pollRef.current) window.clearInterval(pollRef.current);
      };
    }
  }, [tasks, refresh]);

  // ── Actions ──
  const togglePattern = (pat: string) => {
    setSelectedPatterns((prev) =>
      prev.includes(pat) ? prev.filter((p) => p !== pat) : [...prev, pat]
    );
  };

  const start = async () => {
    if (selectedPatterns.length === 0) {
      setError("Select at least one pattern type");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const task = await researchApi.startContinuousResearch({
        scope,
        theme: theme || undefined,
        pattern_types: selectedPatterns,
        lookback_years: lookback,
        similarity_threshold: similarity / 100,
        max_candidates: maxCandidates,
        min_sample_size: minSample,
        min_win_rate: minWinRate / 100,
        min_avg_r: minAvgR,
      });
      setActiveTaskId(task.id);
      await refresh();
    } catch (e: any) {
      setError(e.message || "Failed to start research");
    } finally {
      setBusy(false);
    }
  };

  const stop = async (taskId: string) => {
    setBusy(true);
    try {
      await researchApi.stopTask(taskId);
      await refresh();
    } catch (e: any) {
      setError(e.message || "Failed to stop task");
    } finally {
      setBusy(false);
    }
  };

  const sendToAvt = async (draftId: string) => {
    setBusy(true);
    try {
      await researchApi.sendDraftToAvt(draftId);
      await refresh();
    } catch (e: any) {
      setError(e.message || "Failed to send to AVT");
    } finally {
      setBusy(false);
    }
  };

  const discard = async (draftId: string) => {
    setBusy(true);
    try {
      await researchApi.discardDraft(draftId);
      await refresh();
    } catch (e: any) {
      setError(e.message || "Failed to discard draft");
    } finally {
      setBusy(false);
    }
  };

  const view = async (draftId: string) => {
    try {
      const detail = await researchApi.getDraft(draftId);
      setViewing(detail);
    } catch (e: any) {
      setError(e.message || "Failed to load draft");
    }
  };

  // ── Filter to focus on active task's drafts (or all if no active) ──
  const visibleDrafts = activeTaskId
    ? drafts.filter((d) => d.task_id === activeTaskId || d.status !== "pending_review")
    : drafts;

  const activeTask = tasks.find((t) => t.id === activeTaskId) || tasks[0];
  const progressPct =
    activeTask && activeTask.candidates_planned > 0
      ? Math.round((activeTask.candidates_completed / activeTask.candidates_planned) * 100)
      : 0;

  return (
    <div className="space-y-4">
      {/* ── Start form ── */}
      <div className={CARD}>
        <h3 className="text-sm font-semibold text-[#C4B5FD] mb-3">Start Continuous Research</h3>
        <p className="text-[11px] text-[#64748B] mb-3">
          Background loop tests every (symbol × pattern) combo. Validated strategies become drafts.
          You decide which to send to AVT (virtual trading only — no real money).
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <label className="block text-[10px] text-[#64748B] mb-1 uppercase tracking-wider">Theme (optional)</label>
            <input
              type="text"
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
              placeholder="e.g. momentum on FNO"
              className="w-full bg-[#0A0E1A] border border-[#2A3050] rounded px-2 py-1.5 text-xs text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6]"
            />
          </div>
          <div>
            <label className="block text-[10px] text-[#64748B] mb-1 uppercase tracking-wider">Scope</label>
            <input
              type="text"
              value={scope}
              onChange={(e) => setScope(e.target.value)}
              placeholder="fno | nifty50 | RELIANCE,TCS,..."
              className="w-full bg-[#0A0E1A] border border-[#2A3050] rounded px-2 py-1.5 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#8B5CF6]"
            />
          </div>
        </div>

        <div className="mt-3">
          <label className="block text-[10px] text-[#64748B] mb-1 uppercase tracking-wider">
            Patterns to test ({selectedPatterns.length} selected)
          </label>
          <div className="flex flex-wrap gap-1.5">
            {PATTERN_LIBRARY.map((p) => {
              const on = selectedPatterns.includes(p);
              return (
                <button
                  key={p}
                  onClick={() => togglePattern(p)}
                  className={`text-[11px] px-2 py-1 rounded border transition-colors ${
                    on
                      ? "bg-[#8B5CF6]/20 border-[#8B5CF6] text-[#C4B5FD]"
                      : "border-[#2A3050] text-[#64748B] hover:border-[#475569]"
                  }`}
                >
                  {p.replace(/_/g, " ")}
                </button>
              );
            })}
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3">
          <div>
            <label className="block text-[10px] text-[#64748B] mb-1">Lookback (yrs)</label>
            <select
              value={lookback}
              onChange={(e) => setLookback(Number(e.target.value))}
              className="w-full bg-[#0A0E1A] border border-[#2A3050] rounded px-2 py-1 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#8B5CF6]"
            >
              {[1, 2, 3, 5].map((y) => <option key={y} value={y}>{y}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-[10px] text-[#64748B] mb-1">Similarity %</label>
            <input
              type="number" min={70} max={99} value={similarity}
              onChange={(e) => setSimilarity(Number(e.target.value))}
              className="w-full bg-[#0A0E1A] border border-[#2A3050] rounded px-2 py-1 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#8B5CF6]"
            />
          </div>
          <div>
            <label className="block text-[10px] text-[#64748B] mb-1">Max candidates</label>
            <input
              type="number" min={1} max={500} value={maxCandidates}
              onChange={(e) => setMaxCandidates(Number(e.target.value))}
              className="w-full bg-[#0A0E1A] border border-[#2A3050] rounded px-2 py-1 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#8B5CF6]"
            />
          </div>
        </div>

        <div className="grid grid-cols-3 gap-3 mt-3">
          <div>
            <label className="block text-[10px] text-[#64748B] mb-1">Min sample size</label>
            <input
              type="number" min={10} max={500} value={minSample}
              onChange={(e) => setMinSample(Number(e.target.value))}
              className="w-full bg-[#0A0E1A] border border-[#2A3050] rounded px-2 py-1 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#8B5CF6]"
            />
          </div>
          <div>
            <label className="block text-[10px] text-[#64748B] mb-1">Min win rate %</label>
            <input
              type="number" min={40} max={90} value={minWinRate}
              onChange={(e) => setMinWinRate(Number(e.target.value))}
              className="w-full bg-[#0A0E1A] border border-[#2A3050] rounded px-2 py-1 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#8B5CF6]"
            />
          </div>
          <div>
            <label className="block text-[10px] text-[#64748B] mb-1">Min avg R</label>
            <input
              type="number" step="0.1" min={0} max={3} value={minAvgR}
              onChange={(e) => setMinAvgR(Number(e.target.value))}
              className="w-full bg-[#0A0E1A] border border-[#2A3050] rounded px-2 py-1 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#8B5CF6]"
            />
          </div>
        </div>

        <button
          onClick={start}
          disabled={busy || selectedPatterns.length === 0}
          className="mt-4 w-full bg-[#8B5CF6] hover:bg-[#7C3AED] disabled:opacity-40 text-white text-sm font-semibold py-2 rounded-lg transition-colors"
        >
          {busy ? "Starting..." : "Start Continuous Research"}
        </button>

        {error && (
          <div className="mt-3 bg-[#1C1016] border border-[#EF4444]/30 rounded px-3 py-2 text-xs text-[#FCA5A5]">
            {error}
          </div>
        )}
      </div>

      {/* ── Active task progress ── */}
      {activeTask && (
        <div className={CARD}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-[#F1F5F9]">Task {activeTask.id.slice(0, 12)}</h3>
              <TaskStatusBadge status={activeTask.status} />
            </div>
            {activeTask.status === "running" && (
              <button
                onClick={() => stop(activeTask.id)}
                disabled={busy}
                className="text-xs px-2 py-1 rounded border border-[#EF4444]/40 text-[#FCA5A5] hover:bg-[#EF4444]/10"
              >
                Stop
              </button>
            )}
          </div>

          {activeTask.theme && (
            <p className="text-xs text-[#94A3B8] mb-2">Theme: <span className="text-[#C4B5FD]">{activeTask.theme}</span></p>
          )}

          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-[#94A3B8]">
                {activeTask.candidates_completed} / {activeTask.candidates_planned} candidates
                {activeTask.current_candidate && (
                  <span className="ml-2 text-[#64748B]">· now: {activeTask.current_candidate}</span>
                )}
              </span>
              <span className="font-mono text-[#C4B5FD]">{progressPct}%</span>
            </div>
            <div className="h-2 bg-[#1E2740] rounded-full overflow-hidden">
              <div
                className="h-full bg-[#8B5CF6] transition-all"
                style={{ width: `${progressPct}%` }}
              />
            </div>
            <div className="flex items-center gap-4 text-xs pt-1">
              <span className="text-[#64748B]">
                Validated: <span className="text-[#10B981] font-mono">{activeTask.candidates_validated}</span>
              </span>
              <span className="text-[#64748B]">
                Symbols: {activeTask.symbols.length}
              </span>
              <span className="text-[#64748B]">
                Patterns: {activeTask.pattern_types.length}
              </span>
            </div>
          </div>

          {activeTask.error_message && (
            <div className="mt-3 bg-[#1C1016] border border-[#EF4444]/30 rounded px-3 py-2 text-xs text-[#FCA5A5]">
              Error: {activeTask.error_message}
            </div>
          )}
        </div>
      )}

      {/* ── Drafts list ── */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-[#C4B5FD]">
            Validated Drafts ({visibleDrafts.length})
          </h3>
          {tasks.length > 1 && (
            <select
              value={activeTaskId || ""}
              onChange={(e) => setActiveTaskId(e.target.value || null)}
              className="bg-[#0A0E1A] border border-[#2A3050] rounded px-2 py-1 text-xs text-[#F1F5F9] focus:outline-none focus:border-[#8B5CF6]"
            >
              <option value="">All tasks</option>
              {tasks.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.id.slice(0, 12)} ({t.candidates_validated})
                </option>
              ))}
            </select>
          )}
        </div>

        {visibleDrafts.length === 0 ? (
          <div className={CARD + " text-center text-xs text-[#64748B]"}>
            No drafts yet. Start a research task — drafts will appear here as they're validated.
          </div>
        ) : (
          <div>
            {visibleDrafts.map((d) => (
              <DraftCard
                key={d.id}
                draft={d}
                onSendToAvt={sendToAvt}
                onDiscard={discard}
                onView={view}
                busy={busy}
              />
            ))}
          </div>
        )}
      </div>

      {viewing && <DraftDetailView detail={viewing} onClose={() => setViewing(null)} />}
    </div>
  );
}
