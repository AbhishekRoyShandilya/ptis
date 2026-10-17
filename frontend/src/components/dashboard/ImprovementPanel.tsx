// src/components/dashboard/ImprovementPanel.tsx
import { useState, useEffect, useCallback } from "react";
import { improvementApi } from "@/services/api";

interface Candidate {
  id: string;
  title: string;
  description: string;
  improvement_type: string;
  analysis_layer: string;
  pattern_type: string;
  detection_method: string;
  expected_impact_r: number;
  confidence: number;
  suggested_action: string;
  variant_id: string | null;
  applies_to: string;
  tradeoff?: {
    trades_improved: number;
    trades_hurt: number;
    net_expectancy_change_r: number;
    net_expectancy_change_pct: number;
    probability_net_positive: number;
    runner_impact: string;
    recommendation: string;
    alternatives: { description: string; net_r: number; runner_cost_pct: number }[];
  };
  behavioral_classification?: {
    type: string;
    market_pct: number;
    behavioral_pct: number;
    action: string;
    note: string;
  };
}

interface Experiment {
  experiment_id: string;
  strategy_id: string;
  status: string;
  shadow_trades_count: number;
  trades_needed: number;
  consistency_rate: number;
  avg_r_saved: number;
  false_positive_rate: number;
  ready_to_graduate: boolean;
}

interface Summary {
  strategies_analyzed: number;
  total_candidates: number;
  total_experiments: number;
  by_status: Record<string, number>;
  total_shadow_trades: number;
  total_r_saved: number;
  total_r_missed: number;
}

type SubTab = "overview" | "experiments" | "candidates";

const TYPE_COLORS: Record<string, string> = {
  rr_killer_fix: "#F59E0B",
  early_exit_filter: "#EF4444",
  entry_filter: "#8B5CF6",
  regime_filter: "#3B82F6",
  variant_avoidance: "#EC4899",
  variant_refinement: "#F97316",
  coaching_recommendation: "#6B7280",
};

const TYPE_LABELS: Record<string, string> = {
  rr_killer_fix: "R:R Killer",
  early_exit_filter: "Early Exit",
  entry_filter: "Entry Filter",
  regime_filter: "Regime Filter",
  variant_avoidance: "Variant Avoid",
  variant_refinement: "Variant Refine",
  coaching_recommendation: "Coaching",
};

const LAYER_COLORS: Record<string, string> = {
  regime: "#3B82F6",
  variant: "#EC4899",
  behavioral: "#6B7280",
  pattern: "#A78BFA",
};

const STATUS_COLORS: Record<string, string> = {
  shadow_testing: "#3B82F6",
  validated: "#10B981",
  graduated: "#A78BFA",
  rejected: "#EF4444",
  detected: "#64748B",
  proposed: "#F59E0B",
};

export default function ImprovementPanel() {
  const [subTab, setSubTab] = useState<SubTab>("overview");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [sumData, expData] = await Promise.all([
        improvementApi.summary(),
        improvementApi.experiments(),
      ]);
      setSummary(sumData);
      setExperiments(expData.experiments || []);
    } catch (e: any) {
      setError(e.message || "Failed to load improvement data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 60_000);
    return () => clearInterval(interval);
  }, [refresh]);

  const handleGraduate = async (id: string) => {
    try {
      await improvementApi.graduate(id);
      refresh();
    } catch {}
  };

  const handleReject = async (id: string) => {
    try {
      await improvementApi.reject(id, "Manually rejected");
      refresh();
    } catch {}
  };

  if (loading && !summary) {
    return (
      <div className="flex items-center justify-center h-full text-[#64748B] text-sm">
        Loading improvement lab...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3">
        <p className="text-[#EF4444] text-sm">{error}</p>
        <button
          onClick={refresh}
          className="px-3 py-1.5 text-xs bg-[#1E293B] text-[#94A3B8] rounded hover:bg-[#2A3050]"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Sub-tab bar */}
      <div className="flex items-center gap-1 px-4 py-2 border-b border-[#1E293B] shrink-0">
        {(["overview", "experiments", "candidates"] as SubTab[]).map((t) => (
          <button
            key={t}
            onClick={() => setSubTab(t)}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${
              subTab === t
                ? "bg-[#1E293B] text-[#A78BFA]"
                : "text-[#64748B] hover:text-[#94A3B8]"
            }`}
          >
            {t === "overview" ? "Overview" : t === "experiments" ? "Experiments" : "Candidates"}
          </button>
        ))}
        <button
          onClick={refresh}
          className="ml-auto text-[10px] text-[#475569] hover:text-[#94A3B8] px-2 py-1"
          title="Refresh"
        >
          Refresh
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {subTab === "overview" && <OverviewView summary={summary} experiments={experiments} />}
        {subTab === "experiments" && (
          <ExperimentsView
            experiments={experiments}
            onGraduate={handleGraduate}
            onReject={handleReject}
          />
        )}
        {subTab === "candidates" && <CandidatesView candidates={candidates} summary={summary} onRefresh={refresh} setCandidates={setCandidates} />}
      </div>
    </div>
  );
}

// ── Overview ─────────────────────────────────────────────────────

function OverviewView({ summary, experiments }: { summary: Summary | null; experiments: Experiment[] }) {
  if (!summary) return null;

  const active = experiments.filter((e) => e.status === "shadow_testing");
  const validated = experiments.filter((e) => e.status === "validated" || e.status === "graduated");

  return (
    <div className="space-y-4">
      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-3">
        <StatCard label="Candidates" value={summary.total_candidates} />
        <StatCard label="Experiments" value={summary.total_experiments} />
        <StatCard
          label="R Saved"
          value={`${summary.total_r_saved > 0 ? "+" : ""}${summary.total_r_saved.toFixed(1)}R`}
          color={summary.total_r_saved > 0 ? "#10B981" : "#64748B"}
        />
      </div>

      {/* Status breakdown */}
      {summary.by_status && Object.keys(summary.by_status).length > 0 && (
        <div className="bg-[#0F1629] rounded-lg p-3 border border-[#1E293B]">
          <h4 className="text-[10px] font-semibold text-[#475569] uppercase tracking-wider mb-2">By Status</h4>
          <div className="flex flex-wrap gap-2">
            {Object.entries(summary.by_status).map(([status, count]) => (
              <span
                key={status}
                className="px-2 py-0.5 rounded text-[10px] font-medium"
                style={{
                  backgroundColor: `${STATUS_COLORS[status] || "#475569"}20`,
                  color: STATUS_COLORS[status] || "#475569",
                }}
              >
                {status.replace("_", " ")}: {count}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Active experiments */}
      {active.length > 0 && (
        <div>
          <h4 className="text-[10px] font-semibold text-[#475569] uppercase tracking-wider mb-2">
            Shadow Testing ({active.length})
          </h4>
          {active.slice(0, 5).map((e) => (
            <ExperimentCard key={e.experiment_id} exp={e} />
          ))}
        </div>
      )}

      {/* Validated */}
      {validated.length > 0 && (
        <div>
          <h4 className="text-[10px] font-semibold text-[#475569] uppercase tracking-wider mb-2">
            Validated ({validated.length})
          </h4>
          {validated.map((e) => (
            <ExperimentCard key={e.experiment_id} exp={e} />
          ))}
        </div>
      )}

      {/* Empty state */}
      {summary.total_experiments === 0 && (
        <div className="text-center py-8 text-[#475569] text-sm">
          <p className="mb-1">No improvement experiments yet.</p>
          <p className="text-xs">
            The lab auto-analyzes trades as they close. Need 30+ trades per strategy to start detecting patterns.
          </p>
        </div>
      )}
    </div>
  );
}

// ── Experiments ──────────────────────────────────────────────────

function ExperimentsView({
  experiments,
  onGraduate,
  onReject,
}: {
  experiments: Experiment[];
  onGraduate: (id: string) => void;
  onReject: (id: string) => void;
}) {
  if (experiments.length === 0) {
    return (
      <div className="text-center py-8 text-[#475569] text-sm">
        No experiments yet. Patterns are detected automatically from closed trades.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {experiments.map((e) => (
        <div
          key={e.experiment_id}
          className="bg-[#0F1629] rounded-lg p-3 border border-[#1E293B]"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono text-[#64748B]">{e.experiment_id}</span>
            <span
              className="px-2 py-0.5 rounded text-[10px] font-medium"
              style={{
                backgroundColor: `${STATUS_COLORS[e.status] || "#475569"}20`,
                color: STATUS_COLORS[e.status] || "#475569",
              }}
            >
              {e.status.replace("_", " ")}
            </span>
          </div>

          <div className="text-[11px] text-[#94A3B8] mb-2">
            Strategy: <span className="text-[#CBD5E1]">{e.strategy_id}</span>
          </div>

          {/* Progress bar */}
          <div className="mb-2">
            <div className="flex justify-between text-[10px] text-[#475569] mb-1">
              <span>Shadow trades: {e.shadow_trades_count}</span>
              <span>{e.trades_needed > 0 ? `${e.trades_needed} more needed` : "Ready"}</span>
            </div>
            <div className="h-1.5 bg-[#1E293B] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${Math.min(100, (e.shadow_trades_count / 50) * 100)}%`,
                  backgroundColor: e.ready_to_graduate ? "#10B981" : "#3B82F6",
                }}
              />
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-2 text-[10px]">
            <div>
              <span className="text-[#475569]">Consistency</span>
              <div className={e.consistency_rate >= 0.65 ? "text-[#10B981]" : "text-[#94A3B8]"}>
                {(e.consistency_rate * 100).toFixed(0)}%
              </div>
            </div>
            <div>
              <span className="text-[#475569]">Avg R Saved</span>
              <div className={e.avg_r_saved > 0 ? "text-[#10B981]" : "text-[#94A3B8]"}>
                {e.avg_r_saved.toFixed(2)}R
              </div>
            </div>
            <div>
              <span className="text-[#475569]">False +ve</span>
              <div className={e.false_positive_rate <= 0.2 ? "text-[#10B981]" : "text-[#EF4444]"}>
                {(e.false_positive_rate * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {/* Actions */}
          {(e.status === "validated" || e.ready_to_graduate) && (
            <div className="flex gap-2 mt-3 pt-2 border-t border-[#1E293B]">
              <button
                onClick={() => onGraduate(e.experiment_id)}
                className="flex-1 px-2 py-1 text-[10px] bg-[#10B981]/20 text-[#10B981] rounded hover:bg-[#10B981]/30 transition-colors"
              >
                Graduate
              </button>
              <button
                onClick={() => onReject(e.experiment_id)}
                className="flex-1 px-2 py-1 text-[10px] bg-[#EF4444]/20 text-[#EF4444] rounded hover:bg-[#EF4444]/30 transition-colors"
              >
                Reject
              </button>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ── Candidates ──────────────────────────────────────────────────

function CandidatesView({
  candidates,
  summary,
  onRefresh,
  setCandidates,
}: {
  candidates: Candidate[];
  summary: Summary | null;
  onRefresh: () => void;
  setCandidates: (c: Candidate[]) => void;
}) {
  const [strategyId, setStrategyId] = useState("");
  const [loading, setLoading] = useState(false);

  const loadCandidates = async () => {
    if (!strategyId.trim()) return;
    setLoading(true);
    try {
      const data = await improvementApi.report(strategyId.trim());
      setCandidates(data.candidates || []);
    } catch {
      setCandidates([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-3">
      {/* Strategy selector */}
      <div className="flex gap-2">
        <input
          value={strategyId}
          onChange={(e) => setStrategyId(e.target.value)}
          placeholder="Strategy ID..."
          className="flex-1 px-3 py-1.5 text-xs bg-[#0F1629] border border-[#1E293B] rounded text-[#CBD5E1] placeholder-[#475569] focus:outline-none focus:border-[#A78BFA]/40"
        />
        <button
          onClick={loadCandidates}
          disabled={loading || !strategyId.trim()}
          className="px-3 py-1.5 text-xs bg-[#A78BFA]/20 text-[#A78BFA] rounded hover:bg-[#A78BFA]/30 disabled:opacity-40"
        >
          {loading ? "..." : "Load"}
        </button>
      </div>

      {/* Candidate list */}
      {candidates.length === 0 ? (
        <div className="text-center py-6 text-[#475569] text-xs">
          Enter a strategy ID to view improvement candidates.
        </div>
      ) : (
        candidates.map((c, i) => (
          <div
            key={c.id || i}
            className="bg-[#0F1629] rounded-lg p-3 border border-[#1E293B]"
          >
            {/* Header: type + layer badges */}
            <div className="flex items-start justify-between mb-1">
              <div className="flex gap-1.5">
                <span
                  className="px-1.5 py-0.5 rounded text-[9px] font-medium"
                  style={{
                    backgroundColor: `${TYPE_COLORS[c.improvement_type] || "#475569"}20`,
                    color: TYPE_COLORS[c.improvement_type] || "#475569",
                  }}
                >
                  {TYPE_LABELS[c.improvement_type] || c.improvement_type}
                </span>
                {c.analysis_layer && (
                  <span
                    className="px-1.5 py-0.5 rounded text-[9px]"
                    style={{
                      backgroundColor: `${LAYER_COLORS[c.analysis_layer] || "#475569"}15`,
                      color: LAYER_COLORS[c.analysis_layer] || "#475569",
                    }}
                  >
                    L{c.analysis_layer === "regime" ? "1" : c.analysis_layer === "variant" ? "2" : c.analysis_layer === "behavioral" ? "3" : "4"}
                  </span>
                )}
              </div>
              <span className="text-[10px] text-[#475569]">
                {c.pattern_type === "hybrid" ? "ML + Rules" : c.pattern_type === "fast_layer" ? "Fast ML" : c.pattern_type === "deep_layer" ? "Deep ML" : "Rules"}
              </span>
            </div>

            <h4 className="text-xs text-[#CBD5E1] font-medium mt-1.5 mb-1">{c.title}</h4>
            <p className="text-[11px] text-[#64748B] leading-relaxed mb-2">{c.description}</p>

            {/* Metrics */}
            <div className="flex items-center gap-3 text-[10px] mb-1">
              <span className="text-[#10B981]">Impact: +{c.expected_impact_r}R</span>
              <span className="text-[#475569]">Confidence: {(c.confidence * 100).toFixed(0)}%</span>
              {c.variant_id && (
                <span className="text-[#EC4899]">Variant: {c.variant_id.slice(0, 12)}</span>
              )}
            </div>

            {/* Tradeoff summary */}
            {c.tradeoff && (
              <div className="mt-1.5 p-2 bg-[#0A0E17] rounded border border-[#1E293B]/50 text-[10px]">
                <div className="flex gap-3 mb-1">
                  <span className={c.tradeoff.net_expectancy_change_r > 0 ? "text-[#10B981]" : "text-[#EF4444]"}>
                    Net: {c.tradeoff.net_expectancy_change_r > 0 ? "+" : ""}{c.tradeoff.net_expectancy_change_r}R/trade
                  </span>
                  <span className="text-[#475569]">
                    +{c.tradeoff.trades_improved} improved / -{c.tradeoff.trades_hurt} hurt
                  </span>
                </div>
                <div className="text-[#64748B]">{c.tradeoff.recommendation}</div>
                {c.tradeoff.alternatives && c.tradeoff.alternatives.length > 0 && (
                  <div className="mt-1 text-[#475569]">
                    Alt: {c.tradeoff.alternatives.slice(0, 2).map(a => `${a.description} (${a.net_r > 0 ? "+" : ""}${a.net_r}R)`).join(", ")}
                  </div>
                )}
              </div>
            )}

            {/* Behavioral classification */}
            {c.behavioral_classification && c.behavioral_classification.type !== "market_pattern" && (
              <div className="mt-1.5 p-2 bg-[#F59E0B]/5 rounded border border-[#F59E0B]/20 text-[10px] text-[#F59E0B]">
                {c.behavioral_classification.type === "behavioral_pattern"
                  ? `Behavioral (${(c.behavioral_classification.behavioral_pct * 100).toFixed(0)}%): ${c.behavioral_classification.note}`
                  : `Mixed: ${(c.behavioral_classification.market_pct * 100).toFixed(0)}% market / ${(c.behavioral_classification.behavioral_pct * 100).toFixed(0)}% behavioral`
                }
              </div>
            )}

            {c.suggested_action && (
              <div className="mt-2 pt-2 border-t border-[#1E293B] text-[10px] text-[#8B5CF6]">
                {c.suggested_action}
              </div>
            )}
          </div>
        ))
      )}
    </div>
  );
}

// ── Shared components ────────────────────────────────────────────

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string | number;
  color?: string;
}) {
  return (
    <div className="bg-[#0F1629] rounded-lg p-3 border border-[#1E293B] text-center">
      <div className="text-[10px] text-[#475569] mb-1">{label}</div>
      <div className="text-lg font-semibold" style={{ color: color || "#CBD5E1" }}>
        {value}
      </div>
    </div>
  );
}

function ExperimentCard({ exp }: { exp: Experiment }) {
  const pct = Math.min(100, (exp.shadow_trades_count / 50) * 100);
  return (
    <div className="bg-[#0A0E17] rounded p-2 mb-1.5 border border-[#1E293B]/50">
      <div className="flex items-center justify-between text-[10px]">
        <span className="text-[#94A3B8] font-mono">{exp.experiment_id.slice(0, 16)}</span>
        <span className="text-[#475569]">{exp.shadow_trades_count}/50 trades</span>
      </div>
      <div className="h-1 bg-[#1E293B] rounded-full mt-1 overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            backgroundColor: exp.ready_to_graduate ? "#10B981" : "#3B82F6",
          }}
        />
      </div>
    </div>
  );
}
