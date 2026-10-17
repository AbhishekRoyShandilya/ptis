// src/components/dashboard/WatchlistPanel.tsx
import { useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useCoPlannerStore } from "@/store/coPlannerStore";

export default function WatchlistPanel() {
  const navigate = useNavigate();
  const {
    activeSymbol,
    watchlistStocks,
    watchlistLoading,
    watchlistSearch,
    setActiveSymbol,
    setWatchlistSearch,
    sendMessage,
    loadWatchlist,
    refreshSmartPicks,
  } = useCoPlannerStore();

  useEffect(() => {
    loadWatchlist();
  }, [loadWatchlist]);

  const filtered = useMemo(() => {
    if (!watchlistSearch) return watchlistStocks;
    const q = watchlistSearch.toUpperCase();
    return watchlistStocks.filter(
      (s) =>
        s.symbol.includes(q) ||
        (s.matchReason ?? "").toUpperCase().includes(q)
    );
  }, [watchlistStocks, watchlistSearch]);

  // Grade badge based on matchScore
  const scoreBadge = (score: number) => {
    if (score >= 0.72) return { label: "A", color: "#10B981" };
    if (score >= 0.55) return { label: "B", color: "#F59E0B" };
    return { label: "C", color: "#3B82F6" };
  };

  return (
    <div className="flex flex-col h-full bg-[#0A0E17]">
      {/* Header */}
      <div className="p-3 border-b border-[#2A3050]">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-[#94A3B8]">
              Smart Picks
            </h2>
            {watchlistStocks.length > 0 && (
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#8B5CF6]/15 text-[#A78BFA] font-medium">
                FNO
              </span>
            )}
          </div>
          <button
            onClick={refreshSmartPicks}
            disabled={watchlistLoading}
            className="text-[10px] text-[#8B5CF6] hover:text-[#A78BFA] transition-colors disabled:opacity-40"
            title="Rescan FNO universe for top picks"
          >
            {watchlistLoading ? (
              <span className="inline-block h-3 w-3 border border-[#8B5CF6] border-t-transparent rounded-full animate-spin" />
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 2v6h-6" /><path d="M3 12a9 9 0 0 1 15-6.7L21 8" />
                <path d="M3 22v-6h6" /><path d="M21 12a9 9 0 0 1-15 6.7L3 16" />
              </svg>
            )}
          </button>
        </div>

        {/* Search */}
        <input
          type="text"
          value={watchlistSearch}
          onChange={(e) => setWatchlistSearch(e.target.value)}
          placeholder="Search symbols..."
          className="w-full px-2.5 py-1.5 text-xs bg-[#1A2035] border border-[#2A3050] rounded-lg text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6]/50"
        />
      </div>

      {/* Stock list */}
      <div className="flex-1 overflow-y-auto">
        {watchlistLoading && filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 gap-3">
            <div className="h-5 w-5 border-2 border-[#8B5CF6] border-t-transparent rounded-full animate-spin" />
            <div className="text-[10px] text-[#64748B]">Scanning FNO universe...</div>
          </div>
        ) : filtered.length === 0 ? (
          <div className="text-center py-10 px-4 flex flex-col items-center gap-3">
            <div className="text-[#3A4060] text-3xl">◎</div>
            <div className="text-xs text-[#64748B] leading-relaxed">
              {watchlistSearch
                ? "No matching symbols"
                : "No live market data yet.\nClick Scan to score FNO stocks by\nvolume, momentum & range compression."}
            </div>
            {!watchlistSearch && (
              <button
                onClick={refreshSmartPicks}
                disabled={watchlistLoading}
                className="mt-1 px-4 py-2 text-xs rounded-lg bg-[#8B5CF6]/20 border border-[#8B5CF6]/40 text-[#A78BFA] hover:bg-[#8B5CF6]/30 transition-colors disabled:opacity-40"
              >
                Scan FNO Universe
              </button>
            )}
          </div>
        ) : (
          <div className="divide-y divide-[#2A3050]/50">
            {filtered.map((stock) => {
              const badge = scoreBadge(stock.matchScore);
              return (
                <div key={stock.id} className="group relative">
                  <button
                    onClick={() => setActiveSymbol(activeSymbol === stock.symbol ? null : stock.symbol)}
                    className={`w-full text-left px-3 py-2.5 border-l-2 transition-all duration-150 hover:bg-[#1A2035] ${
                      activeSymbol === stock.symbol
                        ? "border-l-[#8B5CF6] bg-[#8B5CF6]/5"
                        : "border-l-transparent"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-semibold text-[#F1F5F9] font-mono">
                          {stock.symbol}
                        </span>
                        <span
                          className="text-[9px] px-1 py-0.5 rounded font-bold"
                          style={{ color: badge.color, background: `${badge.color}18` }}
                        >
                          {badge.label}
                        </span>
                      </div>
                      {/* Price — hidden on hover to make room for action buttons */}
                      <div className="text-right group-hover:hidden">
                        {stock.lastPrice ? (
                          <>
                            <div className="text-xs font-num text-[#F1F5F9]">
                              ₹{stock.lastPrice.toLocaleString("en-IN")}
                            </div>
                            <div className={`text-[10px] font-num ${(stock.changePct ?? 0) >= 0 ? "text-[#10B981]" : "text-[#EF4444]"}`}>
                              {(stock.changePct ?? 0) >= 0 ? "+" : ""}
                              {(stock.changePct ?? 0).toFixed(2)}%
                            </div>
                          </>
                        ) : (
                          <div className="text-[10px] text-[#3A4060]">--</div>
                        )}
                      </div>
                    </div>

                    {/* Score bar */}
                    <div className="mt-1 flex items-center gap-2">
                      <div className="flex-1 h-1 bg-[#0A0E17] rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${stock.matchScore * 100}%`, background: badge.color }}
                        />
                      </div>
                      <span className="text-[9px] font-num text-[#64748B] w-7 text-right">
                        {(stock.matchScore * 100).toFixed(0)}%
                      </span>
                    </div>

                    {stock.matchReason && (
                      <div className="text-[10px] text-[#64748B] mt-0.5 truncate">
                        {stock.matchReason}
                      </div>
                    )}
                  </button>

                  {/* Hover action buttons — appear top-right on row hover */}
                  <div className="absolute top-2.5 right-2 hidden group-hover:flex gap-1 z-10">
                    {/* Evaluate → Intelligence page, auto-runs eval */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        navigate(`/intelligence?tab=intel&symbol=${stock.symbol}&autorun=1`);
                      }}
                      title="Evaluate setup on Intelligence page"
                      className="flex items-center gap-1 px-2 py-1 text-[10px] font-medium rounded bg-[#8B5CF6]/20 border border-[#8B5CF6]/40 text-[#A78BFA] hover:bg-[#8B5CF6]/35 transition-colors"
                    >
                      <span>◎</span>
                      <span>Eval</span>
                    </button>
                    {/* Co-Plan → stays on dashboard, auto-sends message to ChatPanel */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setActiveSymbol(stock.symbol);
                        sendMessage(`What's the setup on ${stock.symbol} right now? Is there a long-side entry forming?`);
                      }}
                      title="Co-plan this symbol in the chat"
                      className="flex items-center gap-1 px-2 py-1 text-[10px] font-medium rounded bg-[#0EA5E9]/15 border border-[#0EA5E9]/35 text-[#38BDF8] hover:bg-[#0EA5E9]/25 transition-colors"
                    >
                      <span>↗</span>
                      <span>CoP</span>
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-[#2A3050]">
        <div className="text-[10px] text-[#3A4060] text-center">
          {watchlistStocks.length > 0
            ? `${filtered.length} of ${watchlistStocks.length} FNO picks — sorted by conviction`
            : "Scores: volume + momentum + range compression"}
        </div>
      </div>
    </div>
  );
}
