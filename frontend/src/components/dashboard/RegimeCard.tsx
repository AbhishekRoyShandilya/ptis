// src/components/dashboard/RegimeCard.tsx
// Market-regime strip for the Dashboard (moved here from the top navbar).
// Self-populating: if no regime has been detected yet it tries an auto-detect
// (needs a historical market-data feed). When that isn't available it stays
// honest with an "Unknown" state + a manual retry, instead of faking a regime.
import { useCallback, useEffect, useState } from "react";
import { regimeApi } from "@/services/api";

const REGIME_STYLE: { match: (r: string) => boolean; label: string; color: string }[] = [
  { match: (r) => r.includes("trend") || r.includes("bull") || r.includes("bear"), label: "Trending", color: "#10B981" },
  { match: (r) => r.includes("rang") || r.includes("mean"), label: "Ranging", color: "#F59E0B" },
  { match: (r) => r.includes("volatil") || r === "breakout" || r === "breakdown", label: "Volatile", color: "#EF4444" },
  { match: (r) => r.includes("low_vol") || r.includes("quiet") || r.includes("compress"), label: "Quiet", color: "#3B82F6" },
];

function styleFor(regime: string) {
  const r = regime.toLowerCase();
  return REGIME_STYLE.find((s) => s.match(r)) || null;
}

export default function RegimeCard() {
  const [regime, setRegime] = useState<string>("unknown");
  const [confidence, setConfidence] = useState<number | null>(null);
  const [detecting, setDetecting] = useState(false);

  const load = useCallback(async (autoDetect: boolean) => {
    try {
      let r: any = await regimeApi.current();
      // If nothing has populated the regime log yet, try one auto-detect pass.
      if (autoDetect && (!r?.regime || r.regime === "unknown")) {
        setDetecting(true);
        try {
          await regimeApi.detect();
          r = await regimeApi.current();
        } catch {
          /* historical feed unavailable — stay "unknown" */
        } finally {
          setDetecting(false);
        }
      }
      if (r?.regime) setRegime(String(r.regime).toLowerCase());
      setConfidence(typeof r?.confidence === "number" ? r.confidence : null);
    } catch {
      /* leave as unknown */
    }
  }, []);

  useEffect(() => {
    load(true);
  }, [load]);

  const st = styleFor(regime);
  const isUnknown = !st || regime === "unknown";
  const color = st?.color ?? "#64748B";
  const label = st?.label ?? "Unknown";
  const pretty = regime.replace(/_/g, " ");

  return (
    <div className="flex items-center justify-between px-4 py-2 border-b border-[#2A3050] bg-[#0F1629] shrink-0">
      <div className="flex items-center gap-2.5">
        <span className="text-[10px] uppercase tracking-wide text-[#64748B]">Market Regime</span>
        <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg" style={{ background: `${color}1A` }}>
          <span
            className={`h-2 w-2 rounded-full ${detecting ? "animate-pulse" : ""}`}
            style={{ backgroundColor: color }}
          />
          <span className="text-xs font-semibold uppercase tracking-wide" style={{ color }}>
            {detecting ? "Detecting…" : label}
          </span>
          {!isUnknown && !detecting && pretty !== label.toLowerCase() && (
            <span className="text-[10px] text-[#94A3B8] capitalize">{pretty}</span>
          )}
          {confidence != null && !isUnknown && (
            <span className="text-[10px] text-[#64748B]">{Math.round(confidence * 100)}%</span>
          )}
        </div>
      </div>

      {isUnknown && !detecting && (
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-[#64748B] hidden sm:inline">
            No regime detected — needs a historical market-data feed
          </span>
          <button
            onClick={() => load(true)}
            className="text-[10px] px-2 py-1 rounded-md bg-[#1A2035] border border-[#2A3050] text-[#94A3B8] hover:text-[#F1F5F9] transition-colors"
          >
            Detect
          </button>
        </div>
      )}
    </div>
  );
}
