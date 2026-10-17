// src/pages/WatchlistPage.tsx
import { useState, useEffect, useCallback, useRef } from "react";
import { watchlistApi } from "@/services/api";

// ── Types ──────────────────────────────────────────────────────────

interface WatchlistMeta {
  id: string;
  name: string;
  watchlist_type: string;
  strategy_profile_id: string | null;
  strategy_name: string | null;
  description: string | null;
  is_active: boolean;
  auto_scan: boolean;
  item_count: number;
  last_scanned_at: string | null;
}

interface WatchlistItem {
  id: string;
  symbol: string;
  added_by: string;
  status: string;
  match_score: number | null;
  match_reason: string | null;
  last_analysis: Record<string, any> | null;
  added_at: string | null;
}

interface BundleInfo {
  key: string;
  name: string;
  description: string;
  count: number;
  preview: string[];
  live?: boolean;
}

type PageTab = "universe" | "strategies" | "intelligence";

// ── Status tag config ──────────────────────────────────────────────

const STATUS: Record<string, { label: string; color: string; bg: string }> = {
  TRIGGERED:  { label: "TRIGGERED",  color: "#10B981", bg: "rgba(16,185,129,0.15)" },
  READY:      { label: "READY",      color: "#F59E0B", bg: "rgba(245,158,11,0.15)" },
  FORMING:    { label: "FORMING",    color: "#3B82F6", bg: "rgba(59,130,246,0.15)" },
  BREAKDOWN:  { label: "BREAKDOWN",  color: "#EF4444", bg: "rgba(239,68,68,0.15)"  },
  WATCHING:   { label: "WATCHING",   color: "#64748B", bg: "rgba(100,116,139,0.1)" },
};

function getSetupStatus(item: WatchlistItem): string {
  return item.last_analysis?.setup_status ?? "WATCHING";
}

function scoreBadge(score: number | null) {
  if (score === null) return { label: "—", color: "#64748B" };
  if (score >= 0.72) return { label: "A", color: "#10B981" };
  if (score >= 0.55) return { label: "B", color: "#F59E0B" };
  return { label: "C", color: "#3B82F6" };
}

// ── Main page ──────────────────────────────────────────────────────

export default function WatchlistPage() {
  const [tab, setTab] = useState<PageTab>("universe");

  // Master watchlist state
  const [masters, setMasters] = useState<WatchlistMeta[]>([]);
  const [selectedMaster, setSelectedMaster] = useState<WatchlistMeta | null>(null);
  const [masterItems, setMasterItems] = useState<WatchlistItem[]>([]);
  const [symbolInput, setSymbolInput] = useState("");
  const [wlName, setWlName] = useState("");
  const [scanning, setScanning] = useState(false);
  const [loadingItems, setLoadingItems] = useState(false);
  const [creatingWl, setCreatingWl] = useState(false);

  // Bundle state
  const [bundles, setBundles] = useState<BundleInfo[]>([]);
  const [addingBundle, setAddingBundle] = useState<string | null>(null);

  // Strategy watchlists state
  const [strategyWls, setStrategyWls] = useState<WatchlistMeta[]>([]);
  const [strategyItemsMap, setStrategyItemsMap] = useState<Record<string, WatchlistItem[]>>({});
  const [expandedWl, setExpandedWl] = useState<string | null>(null);

  // Polling ref
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Load everything ──────────────────────────────────────────────

  const loadAll = useCallback(async () => {
    try {
      const all = await watchlistApi.list();
      const ms = all.filter((w: WatchlistMeta) => w.watchlist_type === "master");
      const sw = all.filter((w: WatchlistMeta) => w.watchlist_type === "strategy");
      setMasters(ms);
      setStrategyWls(sw);
      if (ms.length > 0 && !selectedMaster) {
        setSelectedMaster(ms[0]);
      }
    } catch {
      /* ignore */
    }
  }, [selectedMaster]);

  const loadBundles = useCallback(async () => {
    try {
      const data = await watchlistApi.bundles();
      setBundles(data);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    loadAll();
    loadBundles();
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  // ── Load master items when selected ─────────────────────────────

  useEffect(() => {
    if (!selectedMaster) { setMasterItems([]); return; }
    setLoadingItems(true);
    watchlistApi.getItems(selectedMaster.id)
      .then(setMasterItems)
      .catch(() => setMasterItems([]))
      .finally(() => setLoadingItems(false));
  }, [selectedMaster]);

  // ── Strategy watchlist polling (30s) ─────────────────────────────

  const loadStrategyItems = useCallback(async (wl: WatchlistMeta) => {
    try {
      const items = await watchlistApi.getItems(wl.id);
      setStrategyItemsMap(prev => ({ ...prev, [wl.id]: items }));
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    if (tab !== "strategies") return;

    // Load immediately
    strategyWls.forEach(loadStrategyItems);

    // Poll every 30s
    pollRef.current = setInterval(() => {
      strategyWls.forEach(loadStrategyItems);
    }, 30_000);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [tab, strategyWls, loadStrategyItems]);

  // ── Actions ───────────────────────────────────────────────────────

  const createMaster = async () => {
    if (!wlName.trim()) return;
    setCreatingWl(true);
    try {
      await watchlistApi.create({ name: wlName, watchlist_type: "master" });
      setWlName("");
      await loadAll();
    } finally {
      setCreatingWl(false);
    }
  };

  const addSymbols = async () => {
    if (!selectedMaster || !symbolInput.trim()) return;
    const syms = symbolInput.split(/[,\s]+/).map(s => s.trim().toUpperCase()).filter(Boolean);
    await watchlistApi.addSymbols(selectedMaster.id, syms);
    setSymbolInput("");
    const items = await watchlistApi.getItems(selectedMaster.id);
    setMasterItems(items);
    loadAll();
  };

  const removeSymbol = async (symbol: string) => {
    if (!selectedMaster) return;
    await watchlistApi.removeSymbol(selectedMaster.id, symbol);
    setMasterItems(prev => prev.filter(i => i.symbol !== symbol));
  };

  const scanWatchlist = async () => {
    if (!selectedMaster) return;
    setScanning(true);
    try {
      await watchlistApi.scan(selectedMaster.id);
      const items = await watchlistApi.getItems(selectedMaster.id);
      setMasterItems(items);
      await loadAll();
    } finally {
      setScanning(false);
    }
  };

  const handleAddBundle = async (bundleKey: string) => {
    if (!selectedMaster) return;
    setAddingBundle(bundleKey);
    try {
      await watchlistApi.addBundle(selectedMaster.id, bundleKey);
      const items = await watchlistApi.getItems(selectedMaster.id);
      setMasterItems(items);
      loadAll();
      loadBundles();
    } catch (e: any) {
      alert(e?.message || "Failed to add bundle");
    } finally {
      setAddingBundle(null);
    }
  };

  // ── Derived data for Intelligence tab ────────────────────────────

  const allStrategyItems: (WatchlistItem & { watchlistName: string })[] = [];
  for (const wl of strategyWls) {
    const items = strategyItemsMap[wl.id] || [];
    for (const item of items) {
      allStrategyItems.push({ ...item, watchlistName: wl.strategy_name || wl.name });
    }
  }
  allStrategyItems.sort((a, b) => (b.match_score ?? 0) - (a.match_score ?? 0));

  // ── Render ────────────────────────────────────────────────────────

  const TABS: { key: PageTab; label: string; icon: string }[] = [
    { key: "universe",     label: "Universe",   icon: "◎" },
    { key: "strategies",   label: "Strategies", icon: "⟐" },
    { key: "intelligence", label: "Intelligence", icon: "💡" },
  ];

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[#F1F5F9]">Watchlist</h1>
          <p className="text-sm text-[#64748B] mt-0.5">
            Universe → Strategy distribution → Real-time setup alerts
          </p>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 bg-[#0F1629] rounded-lg p-1 w-fit">
        {TABS.map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-all flex items-center gap-1.5 ${
              tab === t.key
                ? "bg-[#8B5CF6]/20 text-[#A78BFA]"
                : "text-[#64748B] hover:text-[#94A3B8] hover:bg-[#1A2035]"
            }`}
          >
            <span>{t.icon}</span>
            {t.label}
            {t.key === "strategies" && strategyWls.length > 0 && (
              <span className="text-[10px] bg-[#8B5CF6]/20 text-[#A78BFA] px-1.5 py-0.5 rounded-full">
                {strategyWls.length}
              </span>
            )}
            {t.key === "intelligence" && allStrategyItems.length > 0 && (
              <span className="text-[10px] bg-[#8B5CF6]/20 text-[#A78BFA] px-1.5 py-0.5 rounded-full">
                {allStrategyItems.length}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* ══════ UNIVERSE TAB ══════════════════════════════════════ */}
      {tab === "universe" && (
        <div className="space-y-4">
          {/* Quick-add bundles */}
          <div className="bg-[#0F1629] border border-[#2A3050] rounded-xl p-4">
            <p className="text-xs text-[#64748B] uppercase tracking-wider mb-3 font-medium">
              Quick-add bundles
            </p>
            <div className="flex flex-wrap gap-2">
              {bundles.length === 0 && (
                <div className="text-xs text-[#3A4060]">Loading bundles...</div>
              )}
              {bundles.map(b => (
                <button
                  key={b.key}
                  disabled={addingBundle !== null || !selectedMaster}
                  onClick={() => handleAddBundle(b.key)}
                  className="group flex items-start gap-3 bg-[#1A2035] border border-[#2A3050] rounded-lg px-3 py-2.5 hover:border-[#8B5CF6]/40 hover:bg-[#8B5CF6]/5 transition-all disabled:opacity-50 text-left"
                  title={!selectedMaster ? "Select or create a master watchlist first" : undefined}
                >
                  <div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-semibold text-[#F1F5F9]">{b.name}</span>
                      {b.live && (
                        <span className="text-[9px] bg-[#10B981]/15 text-[#10B981] px-1 rounded">LIVE</span>
                      )}
                    </div>
                    <p className="text-[10px] text-[#64748B] mt-0.5">{b.description}</p>
                    <p className="text-[10px] text-[#3A4060] mt-1">
                      {b.preview.slice(0, 4).join(", ")}{b.count > 4 ? ` +${b.count - 4}` : ""}
                    </p>
                  </div>
                  <div className="ml-auto pl-2 flex-shrink-0">
                    {addingBundle === b.key ? (
                      <span className="inline-block h-3.5 w-3.5 border border-[#8B5CF6] border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <span className="text-[10px] text-[#8B5CF6] group-hover:text-[#A78BFA]">+ Add</span>
                    )}
                  </div>
                </button>
              ))}
            </div>
            {!selectedMaster && masters.length === 0 && (
              <p className="text-[10px] text-[#3A4060] mt-2">Create a master watchlist below to enable quick-add bundles.</p>
            )}
          </div>

          {/* Master watchlist selector */}
          <div className="flex items-center gap-2 flex-wrap">
            {masters.map(wl => (
              <button
                key={wl.id}
                onClick={() => setSelectedMaster(wl)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium border transition-all ${
                  selectedMaster?.id === wl.id
                    ? "bg-[#8B5CF6]/20 border-[#8B5CF6]/50 text-[#A78BFA]"
                    : "bg-[#0F1629] border-[#2A3050] text-[#94A3B8] hover:border-[#8B5CF6]/30"
                }`}
              >
                {wl.name}
                <span className="ml-1.5 text-[10px] text-[#64748B]">({wl.item_count})</span>
              </button>
            ))}

            {/* Create / expand form */}
            {creatingWl ? (
              <div className="flex items-center gap-2">
                <input
                  autoFocus
                  value={wlName}
                  onChange={e => setWlName(e.target.value)}
                  placeholder="Watchlist name..."
                  onKeyDown={e => { if (e.key === "Enter") createMaster(); if (e.key === "Escape") setCreatingWl(false); }}
                  className="px-3 py-1.5 text-sm bg-[#1A2035] border border-[#2A3050] rounded-lg text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6]/50 w-44"
                />
                <button onClick={createMaster} className="px-3 py-1.5 text-sm bg-[#8B5CF6] text-white rounded-lg hover:bg-[#7C3AED]">Create</button>
                <button onClick={() => setCreatingWl(false)} className="text-xs text-[#64748B] hover:text-[#94A3B8]">Cancel</button>
              </div>
            ) : (
              <button
                onClick={() => setCreatingWl(true)}
                className="px-3 py-1.5 rounded-lg text-sm text-[#64748B] hover:text-[#94A3B8] border border-dashed border-[#2A3050] hover:border-[#8B5CF6]/30 transition-all"
              >
                + New
              </button>
            )}
          </div>

          {/* Selected master content */}
          {selectedMaster ? (
            <div className="space-y-3">
              {/* Add symbols bar */}
              <div className="bg-[#0F1629] border border-[#2A3050] rounded-xl p-4">
                <div className="flex gap-2">
                  <input
                    value={symbolInput}
                    onChange={e => setSymbolInput(e.target.value)}
                    placeholder="Add symbols — comma or space separated (e.g. RELIANCE TCS INFY)"
                    onKeyDown={e => e.key === "Enter" && addSymbols()}
                    className="flex-1 bg-[#1A2035] border border-[#2A3050] rounded-lg px-3 py-2 text-sm text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6]/50"
                  />
                  <button
                    onClick={addSymbols}
                    disabled={!symbolInput.trim()}
                    className="px-4 py-2 bg-[#10B981] text-white rounded-lg text-sm font-medium hover:bg-[#059669] disabled:opacity-40"
                  >
                    Add
                  </button>
                  <button
                    onClick={scanWatchlist}
                    disabled={scanning || masterItems.length === 0}
                    className="px-4 py-2 bg-[#8B5CF6] text-white rounded-lg text-sm font-medium hover:bg-[#7C3AED] disabled:opacity-40 flex items-center gap-1.5"
                  >
                    {scanning ? (
                      <>
                        <span className="inline-block h-3 w-3 border border-white border-t-transparent rounded-full animate-spin" />
                        Distributing...
                      </>
                    ) : "Distribute to Strategies"}
                  </button>
                </div>
                {selectedMaster.last_scanned_at && (
                  <p className="text-[10px] text-[#64748B] mt-2">
                    Last distributed: {new Date(selectedMaster.last_scanned_at).toLocaleString("en-IN")}
                  </p>
                )}
              </div>

              {/* Symbol table */}
              <div className="bg-[#0F1629] border border-[#2A3050] rounded-xl overflow-hidden">
                {loadingItems ? (
                  <div className="p-10 text-center text-[#64748B] text-sm">Loading...</div>
                ) : masterItems.length === 0 ? (
                  <div className="p-10 text-center">
                    <div className="text-3xl text-[#3A4060] mb-3">◎</div>
                    <p className="text-sm text-[#64748B]">No symbols yet.</p>
                    <p className="text-xs text-[#3A4060] mt-1">Use a quick-add bundle above or type symbols manually.</p>
                  </div>
                ) : (
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-[#2A3050]">
                        <th className="text-left px-4 py-3 text-[10px] text-[#64748B] uppercase tracking-wider font-medium">Symbol</th>
                        <th className="text-left px-4 py-3 text-[10px] text-[#64748B] uppercase tracking-wider font-medium">Source</th>
                        <th className="text-left px-4 py-3 text-[10px] text-[#64748B] uppercase tracking-wider font-medium">Status</th>
                        <th className="text-left px-4 py-3 text-[10px] text-[#64748B] uppercase tracking-wider font-medium">Match Score</th>
                        <th className="text-left px-4 py-3 text-[10px] text-[#64748B] uppercase tracking-wider font-medium">Reason</th>
                        <th className="px-4 py-3" />
                      </tr>
                    </thead>
                    <tbody>
                      {masterItems.map(item => (
                        <tr key={item.id} className="border-b border-[#2A3050]/40 hover:bg-[#1A2035]/50 transition-colors">
                          <td className="px-4 py-3 font-mono font-semibold text-[#F1F5F9]">{item.symbol}</td>
                          <td className="px-4 py-3 text-xs text-[#64748B]">{item.added_by}</td>
                          <td className="px-4 py-3">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                              item.status === "matched" ? "bg-[#10B981]/15 text-[#10B981]" :
                              item.status === "active"  ? "bg-[#3B82F6]/15 text-[#3B82F6]"  :
                              "bg-[#64748B]/15 text-[#64748B]"
                            }`}>
                              {item.status}
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            {item.match_score != null ? (
                              <div className="flex items-center gap-2">
                                <div className="w-16 h-1 bg-[#1A2035] rounded-full overflow-hidden">
                                  <div
                                    className="h-full rounded-full"
                                    style={{
                                      width: `${item.match_score * 100}%`,
                                      backgroundColor: item.match_score > 0.7 ? "#10B981" : item.match_score > 0.4 ? "#F59E0B" : "#3B82F6",
                                    }}
                                  />
                                </div>
                                <span className="text-xs text-[#94A3B8]">{(item.match_score * 100).toFixed(0)}%</span>
                              </div>
                            ) : (
                              <span className="text-[#3A4060] text-xs">—</span>
                            )}
                          </td>
                          <td className="px-4 py-3 text-xs text-[#94A3B8] max-w-[200px] truncate">{item.match_reason || "—"}</td>
                          <td className="px-4 py-3">
                            <button
                              onClick={() => removeSymbol(item.symbol)}
                              className="text-[#64748B] hover:text-[#EF4444] text-[10px] transition-colors"
                            >
                              Remove
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          ) : (
            masters.length === 0 && (
              <div className="bg-[#0F1629] border border-dashed border-[#2A3050] rounded-xl p-8 text-center">
                <div className="text-3xl text-[#3A4060] mb-3">◎</div>
                <p className="text-sm text-[#94A3B8]">No master watchlist yet.</p>
                <p className="text-xs text-[#64748B] mt-1 mb-4">Create one to start building your universe.</p>
                <button
                  onClick={() => setCreatingWl(true)}
                  className="px-4 py-2 bg-[#8B5CF6]/20 border border-[#8B5CF6]/40 text-[#A78BFA] text-sm rounded-lg hover:bg-[#8B5CF6]/30 transition-colors"
                >
                  Create Master Watchlist
                </button>
              </div>
            )
          )}
        </div>
      )}

      {/* ══════ STRATEGIES TAB ════════════════════════════════════ */}
      {tab === "strategies" && (
        <div className="space-y-4">
          {strategyWls.length === 0 ? (
            <div className="bg-[#0F1629] border border-[#2A3050] rounded-xl p-10 text-center">
              <div className="text-3xl text-[#3A4060] mb-3">⟐</div>
              <p className="text-sm text-[#94A3B8]">No strategy watchlists yet.</p>
              <p className="text-xs text-[#64748B] mt-1">
                Add stocks to your master watchlist, then click "Distribute to Strategies".
                The system creates one watchlist per strategy with matching stocks.
              </p>
            </div>
          ) : (
            strategyWls.map(wl => (
              <StrategySection
                key={wl.id}
                watchlist={wl}
                items={strategyItemsMap[wl.id] || []}
                expanded={expandedWl === wl.id}
                onToggle={() => setExpandedWl(prev => prev === wl.id ? null : wl.id)}
              />
            ))
          )}
        </div>
      )}

      {/* ══════ INTELLIGENCE TAB ══════════════════════════════════ */}
      {tab === "intelligence" && (
        <IntelligenceTab
          items={allStrategyItems}
          onRefreshSmartPicks={async () => {
            await watchlistApi.smartSeed();
            await loadAll();
          }}
        />
      )}
    </div>
  );
}


// ── Strategy Section ───────────────────────────────────────────────

function StrategySection({
  watchlist,
  items,
  expanded,
  onToggle,
}: {
  watchlist: WatchlistMeta;
  items: WatchlistItem[];
  expanded: boolean;
  onToggle: () => void;
}) {
  const triggered = items.filter(i => getSetupStatus(i) === "TRIGGERED").length;
  const ready     = items.filter(i => getSetupStatus(i) === "READY").length;
  const forming   = items.filter(i => getSetupStatus(i) === "FORMING").length;

  return (
    <div className="bg-[#0F1629] border border-[#2A3050] rounded-xl overflow-hidden">
      {/* Header */}
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-4 hover:bg-[#1A2035]/50 transition-colors text-left"
      >
        <div className="flex items-center gap-3">
          <div>
            <h3 className="text-sm font-semibold text-[#F1F5F9]">
              {watchlist.strategy_name || watchlist.name}
            </h3>
            {watchlist.strategy_name && (
              <p className="text-[10px] text-[#64748B]">{watchlist.name}</p>
            )}
          </div>

          {/* Alert counters */}
          <div className="flex items-center gap-1.5">
            {triggered > 0 && (
              <span className="text-[10px] font-bold px-2 py-0.5 rounded" style={{ background: STATUS.TRIGGERED.bg, color: STATUS.TRIGGERED.color }}>
                {triggered} TRIGGERED
              </span>
            )}
            {ready > 0 && (
              <span className="text-[10px] font-bold px-2 py-0.5 rounded" style={{ background: STATUS.READY.bg, color: STATUS.READY.color }}>
                {ready} READY
              </span>
            )}
            {forming > 0 && (
              <span className="text-[10px] font-bold px-2 py-0.5 rounded" style={{ background: STATUS.FORMING.bg, color: STATUS.FORMING.color }}>
                {forming} FORMING
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-xs text-[#64748B]">{items.length} stocks</span>
          <span className="text-[#64748B] text-xs">{expanded ? "▲" : "▼"}</span>
        </div>
      </button>

      {/* Stock rows */}
      {expanded && (
        <div className="border-t border-[#2A3050]">
          {items.length === 0 ? (
            <div className="p-6 text-center text-xs text-[#64748B]">No stocks matched this strategy yet.</div>
          ) : (
            <div className="divide-y divide-[#2A3050]/50">
              {items.map(item => {
                const status = getSetupStatus(item);
                const sc = STATUS[status] || STATUS.WATCHING;
                const badge = scoreBadge(item.match_score);
                const lastSignal = item.last_analysis?.last_signal_type;
                const lastScore = item.last_analysis?.last_signal_score;
                const lastAt = item.last_analysis?.last_signal_at;

                return (
                  <div
                    key={item.id}
                    className="flex items-center px-4 py-2.5 hover:bg-[#1A2035]/40 transition-colors"
                  >
                    {/* Symbol */}
                    <div className="w-28 flex items-center gap-2">
                      <span className="font-mono font-semibold text-sm text-[#F1F5F9]">{item.symbol}</span>
                      <span
                        className="text-[9px] px-1 py-0.5 rounded font-bold"
                        style={{ color: badge.color, background: `${badge.color}18` }}
                      >
                        {badge.label}
                      </span>
                    </div>

                    {/* Status tag */}
                    <div className="w-28">
                      <span
                        className="text-[10px] font-bold px-2 py-0.5 rounded"
                        style={{ background: sc.bg, color: sc.color }}
                      >
                        {sc.label}
                      </span>
                    </div>

                    {/* Last signal info */}
                    <div className="flex-1 text-xs text-[#64748B]">
                      {lastSignal ? (
                        <span>
                          {lastSignal.replace(/_/g, " ")}
                          {lastScore ? ` · ${lastScore.toFixed(0)}/100` : ""}
                          {lastAt ? ` · ${new Date(lastAt).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })}` : ""}
                        </span>
                      ) : (
                        item.match_reason || "—"
                      )}
                    </div>

                    {/* Match score bar */}
                    {item.match_score != null && (
                      <div className="flex items-center gap-2 ml-4">
                        <div className="w-12 h-1 bg-[#0A0E17] rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${item.match_score * 100}%`,
                              backgroundColor: badge.color,
                            }}
                          />
                        </div>
                        <span className="text-[10px] text-[#64748B] w-7 text-right">
                          {(item.match_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}


// ── Intelligence Tab ───────────────────────────────────────────────

function IntelligenceTab({
  items,
  onRefreshSmartPicks,
}: {
  items: (WatchlistItem & { watchlistName: string })[];
  onRefreshSmartPicks: () => Promise<void>;
}) {
  const [filterStatus, setFilterStatus] = useState<string>("ALL");
  const [seeding, setSeeding] = useState(false);

  const statusOptions = ["ALL", "TRIGGERED", "READY", "FORMING", "WATCHING"];

  const filtered = filterStatus === "ALL"
    ? items
    : items.filter(i => getSetupStatus(i) === filterStatus);

  const handleSeed = async () => {
    setSeeding(true);
    try { await onRefreshSmartPicks(); } finally { setSeeding(false); }
  };

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex gap-1 bg-[#0F1629] rounded-lg p-1">
          {statusOptions.map(s => (
            <button
              key={s}
              onClick={() => setFilterStatus(s)}
              className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                filterStatus === s
                  ? "bg-[#8B5CF6]/20 text-[#A78BFA]"
                  : "text-[#64748B] hover:text-[#94A3B8]"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
        <button
          onClick={handleSeed}
          disabled={seeding}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs bg-[#8B5CF6]/15 border border-[#8B5CF6]/30 text-[#A78BFA] hover:bg-[#8B5CF6]/25 transition-colors disabled:opacity-40"
        >
          {seeding ? (
            <span className="inline-block h-3 w-3 border border-[#A78BFA] border-t-transparent rounded-full animate-spin" />
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 2v6h-6" /><path d="M3 12a9 9 0 0 1 15-6.7L21 8" />
              <path d="M3 22v-6h6" /><path d="M21 12a9 9 0 0 1-15 6.7L3 16" />
            </svg>
          )}
          Refresh Smart Picks
        </button>
      </div>

      {filtered.length === 0 ? (
        <div className="bg-[#0F1629] border border-[#2A3050] rounded-xl p-10 text-center">
          <div className="text-3xl text-[#3A4060] mb-3">💡</div>
          <p className="text-sm text-[#94A3B8]">
            {items.length === 0
              ? "No stocks in strategy watchlists yet."
              : `No stocks with status "${filterStatus}".`}
          </p>
        </div>
      ) : (
        <div className="bg-[#0F1629] border border-[#2A3050] rounded-xl overflow-hidden">
          <div className="px-4 py-2 border-b border-[#2A3050] flex items-center justify-between">
            <span className="text-[10px] text-[#64748B] uppercase tracking-wider">
              {filtered.length} stocks ranked by conviction
            </span>
            <span className="text-[10px] text-[#3A4060]">Updated every 30s</span>
          </div>

          <div className="divide-y divide-[#2A3050]/40">
            {filtered.map((item, idx) => {
              const status = getSetupStatus(item);
              const sc = STATUS[status] || STATUS.WATCHING;
              const badge = scoreBadge(item.match_score);

              return (
                <div
                  key={`${item.id}-${idx}`}
                  className="flex items-center px-4 py-3 hover:bg-[#1A2035]/40 transition-colors"
                >
                  {/* Rank */}
                  <div className="w-6 text-[10px] text-[#3A4060] font-num">{idx + 1}</div>

                  {/* Symbol + grade */}
                  <div className="w-32 flex items-center gap-1.5">
                    <span className="font-mono font-bold text-sm text-[#F1F5F9]">{item.symbol}</span>
                    <span
                      className="text-[9px] px-1 py-0.5 rounded font-bold"
                      style={{ color: badge.color, background: `${badge.color}18` }}
                    >
                      {badge.label}
                    </span>
                  </div>

                  {/* Status */}
                  <div className="w-24">
                    <span
                      className="text-[10px] font-bold px-2 py-0.5 rounded"
                      style={{ background: sc.bg, color: sc.color }}
                    >
                      {sc.label}
                    </span>
                  </div>

                  {/* Strategy */}
                  <div className="flex-1 text-xs text-[#64748B] truncate">{item.watchlistName}</div>

                  {/* Reason */}
                  <div className="w-56 text-[11px] text-[#94A3B8] truncate text-right">{item.match_reason || "—"}</div>

                  {/* Score bar */}
                  {item.match_score != null && (
                    <div className="flex items-center gap-2 ml-4">
                      <div className="w-14 h-1 bg-[#0A0E17] rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${item.match_score * 100}%`,
                            backgroundColor: badge.color,
                          }}
                        />
                      </div>
                      <span className="text-[10px] text-[#64748B] w-7 text-right font-num">
                        {(item.match_score * 100).toFixed(0)}%
                      </span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
