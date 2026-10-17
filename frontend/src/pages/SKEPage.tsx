// src/pages/SKEPage.tsx
// Strategy Knowledge Engine — SKO management, signals, scoring playground
import { useState, useEffect, useCallback } from "react";
import { skeApi } from "@/services/api";

type Tab = "list" | "signals" | "scoring";

interface SKOSummary {
  id: string;
  name: string;
  short_code: string;
  status: string;
  tier: number;
  total_trades: number;
  win_rate: number | null;
  expectancy: number | null;
  dqal_grade: string | null;
  tags: string[];
}

interface Signal {
  id: string;
  sko_id: string;
  symbol: string;
  conviction_score: number;
  conviction_band: string;
  sizing_recommendation: string;
  entry_zone: { lower: number; upper: number; sweet_spot: number } | null;
  stop_level: number | null;
  target_levels: number[];
  status: string;
  outcome_r: number | null;
  created_at: string | null;
}

const TIER_BADGES: Record<string, { label: string; color: string }> = {
  hypothesis: { label: "Hypothesis", color: "#6B7280" },
  developing: { label: "Developing", color: "#F59E0B" },
  validated: { label: "Validated", color: "#3B82F6" },
  proven: { label: "Proven", color: "#10B981" },
  archived: { label: "Archived", color: "#9CA3AF" },
};

const GRADE_COLORS: Record<string, string> = {
  A: "#10B981",
  B: "#3B82F6",
  C: "#F59E0B",
  D: "#F97316",
  F: "#EF4444",
};

const BAND_COLORS: Record<string, string> = {
  skip: "#6B7280",
  small: "#F59E0B",
  standard: "#3B82F6",
  aggressive: "#F97316",
  max: "#10B981",
};

export default function SKEPage() {
  const [tab, setTab] = useState<Tab>("list");
  const [skos, setSKOs] = useState<SKOSummary[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Scoring playground state
  const [scoreFeatures, setScoreFeatures] = useState<string>('{"rsi": 35, "compression_ratio": 0.3}');
  const [scoreConfig, setScoreConfig] = useState<string>('{"rsi": {"type": "gaussian", "center": 30, "spread": 10, "weight": 1.5}, "compression_ratio": {"type": "trapezoid", "lower": 0.1, "upper": 0.4, "trap_spread": 0.15, "weight": 1.0}}');
  const [scoreResult, setScoreResult] = useState<any>(null);

  const loadSKOs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await skeApi.list();
      setSKOs(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadSignals = useCallback(async () => {
    setLoading(true);
    try {
      const data = await skeApi.signals();
      setSignals(data);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (tab === "list") loadSKOs();
    if (tab === "signals") loadSignals();
  }, [tab, loadSKOs, loadSignals]);

  const handleBuild = async (id: string) => {
    try {
      await skeApi.build(id);
      loadSKOs();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleValidate = async (id: string) => {
    try {
      const result = await skeApi.validate(id);
      alert(`DQAL: ${result.grade} (${result.overall_score}/100)`);
      loadSKOs();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleScore = async () => {
    try {
      const features = JSON.parse(scoreFeatures);
      const config = JSON.parse(scoreConfig);
      const result = await skeApi.score({ feature_values: features, scoring_config: config });
      setScoreResult(result);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const tabs: { key: Tab; label: string }[] = [
    { key: "list", label: "SKO Library" },
    { key: "signals", label: "Signals" },
    { key: "scoring", label: "Scoring Playground" },
  ];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Strategy Knowledge Engine</h1>
      </div>

      {/* Tab Bar */}
      <div className="flex gap-1 bg-[#1A1F35] rounded-lg p-1">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
              tab === t.key
                ? "bg-[#2A3050] text-white"
                : "text-gray-400 hover:text-white"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-2 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* SKO List */}
      {tab === "list" && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {loading && <p className="text-gray-400 col-span-full">Loading...</p>}
          {!loading && skos.length === 0 && (
            <p className="text-gray-500 col-span-full">No SKOs created yet. Use the API to create one.</p>
          )}
          {skos.map((sko) => {
            const badge = TIER_BADGES[sko.status] || TIER_BADGES.hypothesis;
            const gradeColor = GRADE_COLORS[sko.dqal_grade || "F"];
            return (
              <div
                key={sko.id}
                className="bg-[#1A1F35] border border-[#2A3050] rounded-xl p-5 space-y-3"
              >
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-white font-semibold">{sko.name}</h3>
                    <p className="text-gray-500 text-xs font-mono">{sko.short_code}</p>
                  </div>
                  <span
                    className="px-2 py-0.5 rounded text-xs font-medium"
                    style={{ color: badge.color, backgroundColor: badge.color + "20" }}
                  >
                    {badge.label}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center">
                  <div>
                    <p className="text-gray-500 text-xs">Trades</p>
                    <p className="text-white font-medium">{sko.total_trades}</p>
                  </div>
                  <div>
                    <p className="text-gray-500 text-xs">Win Rate</p>
                    <p className="text-white font-medium">
                      {sko.win_rate != null ? `${(sko.win_rate * 100).toFixed(0)}%` : "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-500 text-xs">DQAL</p>
                    <p className="font-bold" style={{ color: gradeColor }}>
                      {sko.dqal_grade || "—"}
                    </p>
                  </div>
                </div>

                {sko.expectancy != null && (
                  <div className="text-xs text-gray-400">
                    Expectancy: <span className={sko.expectancy > 0 ? "text-green-400" : "text-red-400"}>
                      {sko.expectancy.toFixed(3)}R
                    </span>
                  </div>
                )}

                <div className="flex gap-2 pt-1">
                  <button
                    onClick={() => handleBuild(sko.id)}
                    className="flex-1 px-3 py-1.5 bg-blue-600/20 text-blue-400 rounded text-xs hover:bg-blue-600/30"
                  >
                    Build
                  </button>
                  <button
                    onClick={() => handleValidate(sko.id)}
                    className="flex-1 px-3 py-1.5 bg-purple-600/20 text-purple-400 rounded text-xs hover:bg-purple-600/30"
                  >
                    Validate
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Signals */}
      {tab === "signals" && (
        <div className="space-y-3">
          {loading && <p className="text-gray-400">Loading signals...</p>}
          {!loading && signals.length === 0 && (
            <p className="text-gray-500">No signals generated yet.</p>
          )}
          {signals.map((sig) => {
            const bandColor = BAND_COLORS[sig.conviction_band] || "#6B7280";
            return (
              <div
                key={sig.id}
                className="bg-[#1A1F35] border border-[#2A3050] rounded-lg p-4 flex items-center justify-between"
              >
                <div className="flex items-center gap-4">
                  <div className="text-center">
                    <p className="text-white font-bold text-lg">{sig.symbol}</p>
                    <span
                      className="px-2 py-0.5 rounded text-xs font-medium"
                      style={{ color: bandColor, backgroundColor: bandColor + "20" }}
                    >
                      {sig.conviction_band}
                    </span>
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm">
                      Conviction: <span className="text-white">{(sig.conviction_score * 100).toFixed(0)}%</span>
                    </p>
                    <p className="text-gray-400 text-sm">
                      Size: <span className="text-white">{sig.sizing_recommendation}</span>
                    </p>
                  </div>
                </div>
                <div className="text-right text-sm">
                  {sig.entry_zone && (
                    <p className="text-gray-400">
                      Entry: <span className="text-white">{sig.entry_zone.sweet_spot}</span>
                    </p>
                  )}
                  {sig.stop_level && (
                    <p className="text-gray-400">
                      Stop: <span className="text-red-400">{sig.stop_level}</span>
                    </p>
                  )}
                  <p className="text-gray-500 text-xs mt-1">
                    {sig.created_at ? new Date(sig.created_at).toLocaleString() : ""}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Scoring Playground */}
      {tab === "scoring" && (
        <div className="bg-[#1A1F35] border border-[#2A3050] rounded-xl p-6 space-y-4">
          <h2 className="text-white font-semibold text-lg">Scoring Playground</h2>
          <p className="text-gray-400 text-sm">
            Test fuzzy scoring: enter feature values and scoring config to see conviction output.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-gray-400 text-xs block mb-1">Feature Values (JSON)</label>
              <textarea
                className="w-full bg-[#0F1320] border border-[#2A3050] text-white rounded-lg p-3 font-mono text-sm h-32"
                value={scoreFeatures}
                onChange={(e) => setScoreFeatures(e.target.value)}
              />
            </div>
            <div>
              <label className="text-gray-400 text-xs block mb-1">Scoring Config (JSON)</label>
              <textarea
                className="w-full bg-[#0F1320] border border-[#2A3050] text-white rounded-lg p-3 font-mono text-sm h-32"
                value={scoreConfig}
                onChange={(e) => setScoreConfig(e.target.value)}
              />
            </div>
          </div>
          <button
            onClick={handleScore}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm"
          >
            Compute Conviction
          </button>
          {scoreResult && (
            <div className="bg-[#0F1320] border border-[#2A3050] rounded-lg p-4 space-y-2">
              <div className="flex items-center gap-4">
                <div>
                  <p className="text-gray-500 text-xs">Weighted Score</p>
                  <p className="text-white text-2xl font-bold">
                    {(scoreResult.weighted_score * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs">Band</p>
                  <p
                    className="text-lg font-semibold"
                    style={{ color: BAND_COLORS[scoreResult.conviction_band] || "#fff" }}
                  >
                    {scoreResult.conviction_band}
                  </p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs">Size Multiplier</p>
                  <p className="text-white text-lg">{scoreResult.size_multiplier}x</p>
                </div>
              </div>
              {scoreResult.individual_scores?.length > 0 && (
                <div className="mt-3">
                  <p className="text-gray-500 text-xs mb-2">Individual Scores</p>
                  <div className="space-y-1">
                    {scoreResult.individual_scores.map((s: any) => (
                      <div key={s.feature_name} className="flex justify-between text-sm">
                        <span className="text-gray-400">{s.feature_name}</span>
                        <span className="text-white">
                          {s.raw_value.toFixed(3)} → {(s.normalized_score * 100).toFixed(0)}%
                          <span className="text-gray-500 ml-2">w={s.weight}</span>
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
