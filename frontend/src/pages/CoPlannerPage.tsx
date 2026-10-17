// src/pages/CoPlannerPage.tsx
// Engine-Driven Co-Planner tab components, embedded in the Dashboard co-plan
// (components/dashboard/CoPlanPanel). Exports:
//   PlanningTab    — multi-setup comparison + ranking
//   QuickEvalTab   — single setup conviction + RR assessment
//   RegimeMatrixTab — strategy x regime performance
// (RR Calculator + Data Quality were dropped during the dashboard merge.)

import { useState } from "react";
import {
  coplannerApi,
  type CoPlanSetupInput,
  type CoPlanSessionResult,
  type CoPlanRankedOpportunity,
  type CoPlanEvalResult,
} from "@/services/api";

// ── Colors ──────────────────────────────────────────────────────

const ACTION_COLOR: Record<string, string> = {
  TAKE: "#10B981",
  CAUTIOUS: "#F59E0B",
  SKIP: "#EF4444",
  SKIP_ALL: "#EF4444",
  DO_NOT_TRADE: "#EF4444",
  NO_SETUPS: "#6B7280",
};

const SEVERITY_COLOR: Record<string, string> = {
  HIGH: "#EF4444",
  MEDIUM: "#F59E0B",
  LOW: "#3B82F6",
};

const GRADE_COLOR: Record<string, string> = {
  EXCELLENT: "#10B981",
  GOOD: "#34D399",
  ACCEPTABLE: "#F59E0B",
  POOR: "#EF4444",
  TERRIBLE: "#991B1B",
};

function scoreColor(s: number) {
  if (s >= 0.75) return "#10B981";
  if (s >= 0.60) return "#34D399";
  if (s >= 0.45) return "#F59E0B";
  return "#EF4444";
}

// ── Setup Form Row ──────────────────────────────────────────────

const EMPTY_SETUP: CoPlanSetupInput = {
  symbol: "",
  strategy_name: "",
  timeframe: "15m",
  entry_price: 0,
  stop_loss: 0,
  target_price: 0,
  current_volume: 0,
  avg_volume: 0,
  current_regime: "unknown",
  pattern_score: 0,
};

function SetupRow({
  setup,
  index,
  onChange,
  onRemove,
}: {
  setup: CoPlanSetupInput;
  index: number;
  onChange: (i: number, s: CoPlanSetupInput) => void;
  onRemove: (i: number) => void;
}) {
  const set = (field: string, value: string | number) =>
    onChange(index, { ...setup, [field]: value });

  return (
    <div className="grid grid-cols-[1fr_1fr_80px_90px_90px_90px_120px_32px] gap-2 items-center">
      <input
        className="bg-[#1E293B] border border-[#334155] rounded px-2 py-1 text-sm text-[#F1F5F9]"
        placeholder="Symbol"
        value={setup.symbol}
        onChange={(e) => set("symbol", e.target.value.toUpperCase())}
      />
      <input
        className="bg-[#1E293B] border border-[#334155] rounded px-2 py-1 text-sm text-[#F1F5F9]"
        placeholder="Strategy"
        value={setup.strategy_name}
        onChange={(e) => set("strategy_name", e.target.value)}
      />
      <input
        className="bg-[#1E293B] border border-[#334155] rounded px-2 py-1 text-sm text-[#F1F5F9]"
        placeholder="Entry"
        type="number"
        value={setup.entry_price || ""}
        onChange={(e) => set("entry_price", parseFloat(e.target.value) || 0)}
      />
      <input
        className="bg-[#1E293B] border border-[#334155] rounded px-2 py-1 text-sm text-[#F1F5F9]"
        placeholder="Stop Loss"
        type="number"
        value={setup.stop_loss || ""}
        onChange={(e) => set("stop_loss", parseFloat(e.target.value) || 0)}
      />
      <input
        className="bg-[#1E293B] border border-[#334155] rounded px-2 py-1 text-sm text-[#F1F5F9]"
        placeholder="Target"
        type="number"
        value={setup.target_price || ""}
        onChange={(e) => set("target_price", parseFloat(e.target.value) || 0)}
      />
      <input
        className="bg-[#1E293B] border border-[#334155] rounded px-2 py-1 text-sm text-[#F1F5F9]"
        placeholder="Volume"
        type="number"
        value={setup.current_volume || ""}
        onChange={(e) => set("current_volume", parseFloat(e.target.value) || 0)}
      />
      <select
        className="bg-[#1E293B] border border-[#334155] rounded px-2 py-1 text-sm text-[#F1F5F9]"
        value={setup.current_regime}
        onChange={(e) => set("current_regime", e.target.value)}
      >
        <option value="unknown">Unknown</option>
        <option value="trending_up">Trending Up</option>
        <option value="trending_down">Trending Down</option>
        <option value="ranging">Ranging</option>
        <option value="high_volatility">High Volatility</option>
        <option value="low_volatility">Low Volatility</option>
        <option value="breakout">Breakout</option>
        <option value="mean_reverting">Mean Reverting</option>
      </select>
      <button
        onClick={() => onRemove(index)}
        className="text-[#EF4444] hover:text-red-300 text-lg font-bold"
        title="Remove"
      >
        x
      </button>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
//  TAB 1: Planning Session
// ══════════════════════════════════════════════════════════════════

export function PlanningTab() {
  const [setups, setSetups] = useState<CoPlanSetupInput[]>([{ ...EMPTY_SETUP }]);
  const [result, setResult] = useState<CoPlanSessionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateSetup = (i: number, s: CoPlanSetupInput) => {
    const next = [...setups];
    next[i] = s;
    setSetups(next);
  };

  const removeSetup = (i: number) => {
    if (setups.length <= 1) return;
    setSetups(setups.filter((_, idx) => idx !== i));
  };

  const addSetup = () => setSetups([...setups, { ...EMPTY_SETUP }]);

  const runPlan = async () => {
    const valid = setups.filter((s) => s.symbol && s.strategy_name);
    if (valid.length === 0) {
      setError("Add at least one setup with symbol and strategy");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await coplannerApi.plan(valid);
      setResult(data);
    } catch (e: any) {
      setError(e.message || "Planning failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Setup inputs */}
      <div className="bg-[#0F172A] border border-[#1E293B] rounded-lg p-4 space-y-3">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-[#94A3B8]">SETUPS TO EVALUATE</h3>
          <button
            onClick={addSetup}
            className="text-xs bg-[#1E293B] text-[#94A3B8] hover:text-[#F1F5F9] px-3 py-1 rounded border border-[#334155]"
          >
            + Add Setup
          </button>
        </div>
        <div className="grid grid-cols-[1fr_1fr_80px_90px_90px_90px_120px_32px] gap-2 text-xs text-[#64748B] mb-1">
          <span>Symbol</span><span>Strategy</span><span>Entry</span>
          <span>SL</span><span>Target</span><span>Volume</span><span>Regime</span><span></span>
        </div>
        {setups.map((s, i) => (
          <SetupRow key={i} setup={s} index={i} onChange={updateSetup} onRemove={removeSetup} />
        ))}
        <button
          onClick={runPlan}
          disabled={loading}
          className="mt-3 px-6 py-2 bg-[#6366F1] hover:bg-[#4F46E5] disabled:opacity-50 text-white text-sm font-medium rounded"
        >
          {loading ? "Computing..." : "Run Planning Session"}
        </button>
      </div>

      {error && <div className="text-[#EF4444] text-sm">{error}</div>}

      {/* Result */}
      {result && (
        <div className="space-y-4">
          {/* Behavioral check */}
          <div className="bg-[#0F172A] border border-[#1E293B] rounded-lg p-4">
            <h3 className="text-sm font-semibold text-[#94A3B8] mb-2">BEHAVIORAL CHECK</h3>
            <div className="flex items-center gap-3 mb-2">
              <span
                className="text-sm font-medium"
                style={{ color: result.behavioral_check.proceed ? "#10B981" : "#EF4444" }}
              >
                {result.behavioral_check.proceed ? "CLEAR" : "BLOCKED"}
              </span>
              <span className="text-xs text-[#64748B]">
                Risk: {result.behavioral_check.risk_score}/{result.behavioral_check.max_risk_score}
              </span>
            </div>
            <p className="text-sm text-[#CBD5E1]">{result.behavioral_check.verdict}</p>
            {result.behavioral_check.warnings.length > 0 && (
              <div className="mt-2 space-y-1">
                {result.behavioral_check.warnings.map((w, i) => (
                  <div key={i} className="text-xs flex gap-2">
                    <span style={{ color: SEVERITY_COLOR[w.severity] || "#6B7280" }}>
                      [{w.severity}]
                    </span>
                    <span className="text-[#CBD5E1]">{w.message}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Recommendation */}
          <div className="bg-[#0F172A] border border-[#1E293B] rounded-lg p-4">
            <h3 className="text-sm font-semibold text-[#94A3B8] mb-2">RECOMMENDATION</h3>
            <div className="flex items-center gap-3">
              <span
                className="text-lg font-bold"
                style={{ color: ACTION_COLOR[result.recommendation.action] || "#6B7280" }}
              >
                {result.recommendation.action}
              </span>
              {result.recommendation.primary && (
                <span className="text-[#F1F5F9] font-medium">{result.recommendation.primary}</span>
              )}
              {result.recommendation.sizing && (
                <span className="text-xs text-[#64748B] bg-[#1E293B] px-2 py-0.5 rounded">
                  {result.recommendation.sizing}
                </span>
              )}
            </div>
            {result.recommendation.message && (
              <p className="text-sm text-[#CBD5E1] mt-1">{result.recommendation.message}</p>
            )}
          </div>

          {/* Ranked Opportunities */}
          {result.ranked_opportunities.length > 0 && (
            <div className="bg-[#0F172A] border border-[#1E293B] rounded-lg p-4">
              <h3 className="text-sm font-semibold text-[#94A3B8] mb-3">
                RANKED OPPORTUNITIES ({result.actionable_count} actionable / {result.total_scanned} scanned)
              </h3>
              <div className="space-y-3">
                {result.ranked_opportunities.map((opp) => (
                  <OpportunityCard key={opp.rank} opp={opp} />
                ))}
              </div>
            </div>
          )}

          {/* Regime Warnings */}
          {result.regime_warnings.length > 0 && (
            <div className="bg-[#0F172A] border border-[#1E293B] rounded-lg p-4">
              <h3 className="text-sm font-semibold text-[#94A3B8] mb-2">REGIME WARNINGS</h3>
              {result.regime_warnings.map((w, i) => (
                <div key={i} className="text-sm text-[#F59E0B] mb-1">
                  {w.symbol} ({w.strategy} in {w.regime}): {w.warning}
                </div>
              ))}
            </div>
          )}

          {/* LLM Explanation */}
          {result.explanation && (
            <div className="bg-[#0F172A] border border-[#1E293B] rounded-lg p-4">
              <h3 className="text-sm font-semibold text-[#94A3B8] mb-2">ENGINE SUMMARY</h3>
              <p className="text-sm text-[#CBD5E1] whitespace-pre-line">{result.explanation}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function OpportunityCard({ opp }: { opp: CoPlanRankedOpportunity }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div
      className="border border-[#1E293B] rounded p-3 cursor-pointer hover:border-[#334155]"
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-xs text-[#64748B] w-5">#{opp.rank}</span>
          <span className="font-medium text-[#F1F5F9]">{opp.symbol}</span>
          <span className="text-xs text-[#94A3B8]">{opp.strategy_name}</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-sm font-mono" style={{ color: scoreColor(opp.composite_score) }}>
            {(opp.composite_score * 100).toFixed(1)}%
          </span>
          <span
            className="text-xs font-medium px-2 py-0.5 rounded"
            style={{
              color: ACTION_COLOR[opp.action] || "#6B7280",
              background: `${ACTION_COLOR[opp.action] || "#6B7280"}20`,
            }}
          >
            {opp.action}
          </span>
          {opp.sizing > 0 && (
            <span className="text-xs text-[#64748B]">{(opp.sizing * 100).toFixed(0)}% size</span>
          )}
        </div>
      </div>
      <p className="text-xs text-[#94A3B8] mt-1">{opp.explanation}</p>

      {expanded && (
        <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
          {/* Conviction breakdown */}
          <div className="space-y-1">
            <div className="text-[#64748B] font-semibold">Conviction Components</div>
            {Object.entries(opp.conviction.components).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-[#94A3B8]">{k.replace(/_/g, " ")}</span>
                <span style={{ color: scoreColor(v) }}>{(v * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
          {/* Score breakdown */}
          <div className="space-y-1">
            <div className="text-[#64748B] font-semibold">Ranking Factors</div>
            {Object.entries(opp.score_breakdown).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-[#94A3B8]">{k.replace(/_/g, " ")}</span>
                <span style={{ color: scoreColor(v) }}>{(v * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
          {/* RR Assessment */}
          {opp.rr_assessment && (
            <div className="col-span-2 flex gap-4 mt-1 border-t border-[#1E293B] pt-2">
              <span className="text-[#94A3B8]">RR: {opp.rr_assessment.expected_rr.toFixed(2)}</span>
              <span style={{ color: GRADE_COLOR[opp.rr_assessment.grade] || "#6B7280" }}>
                {opp.rr_assessment.grade}
              </span>
              <span className="text-[#94A3B8]">{opp.rr_assessment.message}</span>
            </div>
          )}
          {/* Warnings */}
          {opp.conviction.warnings.length > 0 && (
            <div className="col-span-2 space-y-1 border-t border-[#1E293B] pt-2">
              {opp.conviction.warnings.map((w, i) => (
                <div key={i} className="flex gap-2">
                  <span style={{ color: SEVERITY_COLOR[w.severity] || "#6B7280" }}>
                    [{w.type}]
                  </span>
                  <span className="text-[#CBD5E1]">{w.message}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
//  TAB 2: Quick Eval
// ══════════════════════════════════════════════════════════════════

export function QuickEvalTab() {
  const [setup, setSetup] = useState<CoPlanSetupInput>({ ...EMPTY_SETUP });
  const [result, setResult] = useState<CoPlanEvalResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = (field: string, value: string | number) =>
    setSetup((prev) => ({ ...prev, [field]: value }));

  const run = async () => {
    if (!setup.symbol || !setup.strategy_name) {
      setError("Symbol and strategy required");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await coplannerApi.evaluate(setup);
      setResult(data);
    } catch (e: any) {
      setError(e.message || "Evaluation failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="bg-[#0F172A] border border-[#1E293B] rounded-lg p-4">
        <h3 className="text-sm font-semibold text-[#94A3B8] mb-3">SINGLE SETUP</h3>
        <div className="grid grid-cols-4 gap-3">
          <input className="bg-[#1E293B] border border-[#334155] rounded px-2 py-1.5 text-sm text-[#F1F5F9]"
            placeholder="Symbol" value={setup.symbol}
            onChange={(e) => set("symbol", e.target.value.toUpperCase())} />
          <input className="bg-[#1E293B] border border-[#334155] rounded px-2 py-1.5 text-sm text-[#F1F5F9]"
            placeholder="Strategy" value={setup.strategy_name}
            onChange={(e) => set("strategy_name", e.target.value)} />
          <input className="bg-[#1E293B] border border-[#334155] rounded px-2 py-1.5 text-sm text-[#F1F5F9]"
            placeholder="Entry Price" type="number" value={setup.entry_price || ""}
            onChange={(e) => set("entry_price", parseFloat(e.target.value) || 0)} />
          <select className="bg-[#1E293B] border border-[#334155] rounded px-2 py-1.5 text-sm text-[#F1F5F9]"
            value={setup.current_regime} onChange={(e) => set("current_regime", e.target.value)}>
            <option value="unknown">Unknown Regime</option>
            <option value="trending_up">Trending Up</option>
            <option value="trending_down">Trending Down</option>
            <option value="ranging">Ranging</option>
            <option value="high_volatility">High Volatility</option>
            <option value="low_volatility">Low Volatility</option>
          </select>
          <input className="bg-[#1E293B] border border-[#334155] rounded px-2 py-1.5 text-sm text-[#F1F5F9]"
            placeholder="Stop Loss" type="number" value={setup.stop_loss || ""}
            onChange={(e) => set("stop_loss", parseFloat(e.target.value) || 0)} />
          <input className="bg-[#1E293B] border border-[#334155] rounded px-2 py-1.5 text-sm text-[#F1F5F9]"
            placeholder="Target" type="number" value={setup.target_price || ""}
            onChange={(e) => set("target_price", parseFloat(e.target.value) || 0)} />
          <input className="bg-[#1E293B] border border-[#334155] rounded px-2 py-1.5 text-sm text-[#F1F5F9]"
            placeholder="Volume" type="number" value={setup.current_volume || ""}
            onChange={(e) => set("current_volume", parseFloat(e.target.value) || 0)} />
          <input className="bg-[#1E293B] border border-[#334155] rounded px-2 py-1.5 text-sm text-[#F1F5F9]"
            placeholder="Avg Volume" type="number" value={setup.avg_volume || ""}
            onChange={(e) => set("avg_volume", parseFloat(e.target.value) || 0)} />
        </div>
        <button onClick={run} disabled={loading}
          className="mt-3 px-6 py-2 bg-[#6366F1] hover:bg-[#4F46E5] disabled:opacity-50 text-white text-sm font-medium rounded">
          {loading ? "Evaluating..." : "Evaluate Setup"}
        </button>
      </div>

      {error && <div className="text-[#EF4444] text-sm">{error}</div>}

      {result && (
        <div className="grid grid-cols-2 gap-4">
          {/* Conviction */}
          <div className="bg-[#0F172A] border border-[#1E293B] rounded-lg p-4">
            <h3 className="text-sm font-semibold text-[#94A3B8] mb-2">CONVICTION</h3>
            <div className="flex items-baseline gap-3 mb-2">
              <span className="text-2xl font-bold" style={{ color: scoreColor(result.conviction.score) }}>
                {(result.conviction.score * 100).toFixed(1)}%
              </span>
              <span className="text-sm text-[#94A3B8]">{result.conviction.level}</span>
            </div>
            <p className="text-sm text-[#CBD5E1] mb-3">{result.conviction.verdict}</p>
            <div className="space-y-1">
              {Object.entries(result.conviction.components).map(([k, v]) => (
                <div key={k} className="flex justify-between text-xs">
                  <span className="text-[#94A3B8]">{k.replace(/_/g, " ")}</span>
                  <span style={{ color: scoreColor(v) }}>{(v * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
            <div className="mt-2 text-xs text-[#64748B]">
              Sizing: {(result.conviction.sizing * 100).toFixed(0)}%
            </div>
          </div>

          {/* Regime + Avoidance */}
          <div className="bg-[#0F172A] border border-[#1E293B] rounded-lg p-4">
            <h3 className="text-sm font-semibold text-[#94A3B8] mb-2">REGIME STATS</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-[#94A3B8]">Win Rate</span>
                <span style={{ color: scoreColor(result.regime.win_rate) }}>
                  {(result.regime.win_rate * 100).toFixed(0)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#94A3B8]">Avg RR</span>
                <span className="text-[#F1F5F9]">{result.regime.avg_rr.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#94A3B8]">Sample</span>
                <span className="text-[#F1F5F9]">{result.regime.trades} trades ({result.regime.confidence})</span>
              </div>
            </div>
            {result.avoidance.avoid && (
              <div className="mt-3 p-2 bg-[#7F1D1D20] border border-[#7F1D1D] rounded text-xs text-[#EF4444]">
                AVOID: {result.avoidance.reason}
              </div>
            )}
          </div>

          {/* RR Assessment */}
          {result.rr && (
            <div className="bg-[#0F172A] border border-[#1E293B] rounded-lg p-4 col-span-2">
              <h3 className="text-sm font-semibold text-[#94A3B8] mb-2">RISK-REWARD</h3>
              <div className="flex items-center gap-4">
                <span className="text-xl font-bold text-[#F1F5F9]">{result.rr.expected_rr.toFixed(2)}R</span>
                <span style={{ color: GRADE_COLOR[result.rr.grade] || "#6B7280" }} className="font-medium">
                  {result.rr.grade}
                </span>
                <span className={result.rr.acceptable ? "text-[#10B981]" : "text-[#EF4444]"}>
                  {result.rr.acceptable ? "Acceptable" : "Below threshold"}
                </span>
              </div>
              <p className="text-sm text-[#CBD5E1] mt-1">{result.rr.message}</p>
            </div>
          )}

          {/* Warnings */}
          {result.conviction.warnings.length > 0 && (
            <div className="col-span-2 space-y-1">
              {result.conviction.warnings.map((w, i) => (
                <div key={i} className="text-xs flex gap-2">
                  <span style={{ color: SEVERITY_COLOR[w.severity] || "#6B7280" }}>[{w.type}]</span>
                  <span className="text-[#CBD5E1]">{w.message}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
//  TAB 3: Regime Matrix
// ══════════════════════════════════════════════════════════════════

export function RegimeMatrixTab() {
  const [matrix, setMatrix] = useState<Record<string, any> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rebuilding, setRebuilding] = useState(false);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await coplannerApi.regimeMatrix();
      setMatrix(data);
    } catch (e: any) {
      setError(e.message || "Failed to load matrix");
    } finally {
      setLoading(false);
    }
  };

  const rebuild = async () => {
    setRebuilding(true);
    setError(null);
    try {
      await coplannerApi.rebuildMatrix();
      await load();
    } catch (e: any) {
      setError(e.message || "Rebuild failed");
    } finally {
      setRebuilding(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button onClick={load} disabled={loading}
          className="px-4 py-2 bg-[#6366F1] hover:bg-[#4F46E5] disabled:opacity-50 text-white text-sm rounded">
          {loading ? "Loading..." : "Load Matrix"}
        </button>
        <button onClick={rebuild} disabled={rebuilding}
          className="px-4 py-2 bg-[#1E293B] hover:bg-[#334155] disabled:opacity-50 text-[#94A3B8] text-sm rounded border border-[#334155]">
          {rebuilding ? "Rebuilding..." : "Force Rebuild"}
        </button>
      </div>

      {error && <div className="text-[#EF4444] text-sm">{error}</div>}

      {matrix && (
        <div className="bg-[#0F172A] border border-[#1E293B] rounded-lg p-4">
          <h3 className="text-sm font-semibold text-[#94A3B8] mb-3">STRATEGY x REGIME MATRIX</h3>
          {matrix.strategies && Array.isArray(matrix.strategies) ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-[#64748B] text-xs">
                    <th className="text-left py-2 pr-4">Strategy</th>
                    <th className="text-left py-2 pr-4">Regime</th>
                    <th className="text-right py-2 pr-4">WR</th>
                    <th className="text-right py-2 pr-4">Avg RR</th>
                    <th className="text-right py-2 pr-4">Trades</th>
                    <th className="text-right py-2">Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {matrix.strategies.map((row: any, i: number) => (
                    <tr key={i} className="border-t border-[#1E293B]">
                      <td className="py-1.5 pr-4 text-[#F1F5F9]">{row.strategy}</td>
                      <td className="py-1.5 pr-4 text-[#94A3B8]">{row.regime}</td>
                      <td className="py-1.5 pr-4 text-right" style={{ color: scoreColor(row.win_rate || 0) }}>
                        {((row.win_rate || 0) * 100).toFixed(0)}%
                      </td>
                      <td className="py-1.5 pr-4 text-right text-[#F1F5F9]">
                        {(row.avg_rr || 0).toFixed(2)}
                      </td>
                      <td className="py-1.5 pr-4 text-right text-[#94A3B8]">{row.total_trades || 0}</td>
                      <td className="py-1.5 text-right text-[#64748B]">{row.confidence || "N/A"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <pre className="text-xs text-[#94A3B8] overflow-auto max-h-96">
              {JSON.stringify(matrix, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
