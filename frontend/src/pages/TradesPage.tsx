// src/pages/TradesPage.tsx
import { useEffect, useState } from "react";
import { tradesApi, type Trade } from "@/services/api";
import { TradeEntryDialog } from "@/components/trade/TradeEntryDialog";
import { TradeCloseDialog } from "@/components/trade/TradeCloseDialog";

function getStatusBadge(status: string) {
  if (status === "open") return "bg-[#10B981]/10 text-[#10B981]";
  if (status === "closed") return "bg-[#3B82F6]/10 text-[#3B82F6]";
  return "bg-[#64748B]/10 text-[#64748B]";
}

function TradeRow({ trade, onClose }: { trade: Trade; onClose: (id: string) => void }) {
  const rColor =
    (trade.r_multiple ?? 0) >= 1
      ? "#10B981"
      : (trade.r_multiple ?? 0) >= 0
      ? "#F59E0B"
      : "#EF4444";

  return (
    <div className="flex items-center gap-3 py-3 border-b border-[#2A3050] last:border-0 hover:bg-[#1A2035]/50 rounded px-2 transition-colors group animate-fade-in">
      {/* Direction */}
      <span
        className={`text-[10px] px-1.5 py-0.5 rounded font-semibold w-12 text-center ${
          trade.direction === "long"
            ? "bg-[#10B981]/10 text-[#10B981]"
            : "bg-[#EF4444]/10 text-[#EF4444]"
        }`}
      >
        {trade.direction.toUpperCase()}
      </span>

      {/* Symbol + strategy */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold text-[#F1F5F9]">{trade.symbol}</span>
          {trade.resolved_strategy && (
            <span className="text-[10px] text-[#64748B] hidden sm:inline">
              {trade.resolved_strategy}
            </span>
          )}
        </div>
        <div className="text-[10px] text-[#64748B] mt-0.5">
          {new Date(trade.entry_time).toLocaleString("en-IN", {
            dateStyle: "short",
            timeStyle: "short",
          })}
          {trade.market_regime && ` · ${trade.market_regime}`}
        </div>
      </div>

      {/* Entry / Exit */}
      <div className="hidden md:block text-right">
        <div className="text-xs font-num text-[#94A3B8]">
          ₹{trade.entry_price.toLocaleString()}
        </div>
        {trade.exit_price && (
          <div className="text-[10px] font-num text-[#64748B]">
            → ₹{trade.exit_price.toLocaleString()}
          </div>
        )}
      </div>

      {/* R-Multiple */}
      {trade.r_multiple !== null && (
        <div
          className="text-sm font-bold font-num w-14 text-right"
          style={{ color: rColor }}
        >
          {trade.r_multiple >= 0 ? "+" : ""}{trade.r_multiple.toFixed(2)}R
        </div>
      )}

      {/* P&L — net, with gross + charges breakdown when charges exist */}
      {trade.pnl !== null && (() => {
        const charges = trade.commission ?? 0;
        const net = trade.pnl ?? 0;
        const gross = net + charges;   // pnl is stored NET → gross = net + charges
        const fmt = (v: number) =>
          `${v >= 0 ? "+" : "-"}₹${Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
        if (charges > 0) {
          return (
            <div className="w-28 text-right font-num leading-tight">
              <div className="text-[10px] text-[#94A3B8]">P&amp;L {fmt(gross)}</div>
              <div className="text-[10px] text-[#F59E0B]">Charges -₹{charges.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
              <div className="text-sm font-bold" style={{ color: net >= 0 ? "#10B981" : "#EF4444" }}>
                Net {fmt(net)}
              </div>
            </div>
          );
        }
        return (
          <div
            className="text-sm font-bold font-num w-28 text-right"
            style={{ color: net >= 0 ? "#10B981" : "#EF4444" }}
          >
            {fmt(net)}
          </div>
        );
      })()}

      {/* Discipline */}
      {trade.discipline_score !== null && (
        <div className="hidden lg:flex items-center gap-1.5">
          <div className="w-10 h-1 bg-[#0A0E17] rounded-full overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{
                width: `${(trade.discipline_score ?? 0) * 10}%`,
                backgroundColor:
                  (trade.discipline_score ?? 0) >= 7
                    ? "#10B981"
                    : (trade.discipline_score ?? 0) >= 5
                    ? "#F59E0B"
                    : "#EF4444",
              }}
            />
          </div>
          <span className="text-[10px] font-num text-[#64748B]">
            {trade.discipline_score?.toFixed(1)}
          </span>
        </div>
      )}

      {/* Status + Close */}
      <div className="flex items-center gap-2">
        <span className={`text-[10px] px-1.5 py-0.5 rounded ${getStatusBadge(trade.status)}`}>
          {trade.status}
        </span>
        {trade.status === "open" && (
          <button
            onClick={() => onClose(trade.id)}
            className="opacity-0 group-hover:opacity-100 transition-opacity text-[10px] px-2 py-1 rounded bg-[#8B5CF6]/10 text-[#A78BFA] hover:bg-[#8B5CF6]/20"
          >
            Close
          </button>
        )}
      </div>
    </div>
  );
}

export default function TradesPage() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterStatus, setFilterStatus] = useState<"all" | "open" | "closed">("all");
  const [entryOpen, setEntryOpen] = useState(false);
  const [closeTradeId, setCloseTradeId] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [importMsg, setImportMsg] = useState<string | null>(null);

  const fetchTrades = () => {
    const params = filterStatus !== "all" ? { status: filterStatus } : undefined;
    tradesApi.list(params).then(setTrades).catch(console.error).finally(() => setLoading(false));
  };

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setImporting(true);
    setImportMsg(null);
    try {
      const r = await tradesApi.importPnl(f);
      setImportMsg(`Imported ${r.imported} trades (${r.duplicates} already present, ${r.skipped} skipped).`);
      fetchTrades();
    } catch (err: unknown) {
      setImportMsg(err instanceof Error ? err.message : "Import failed");
    } finally {
      setImporting(false);
      e.target.value = "";
    }
  };

  useEffect(() => {
    fetchTrades();
  }, [filterStatus]);

  const closedTrades = trades.filter((t) => t.status === "closed");
  const totalPnl = closedTrades.reduce((s, t) => s + (t.pnl ?? 0), 0);
  const winRate = closedTrades.length
    ? (closedTrades.filter((t) => (t.pnl ?? 0) > 0).length / closedTrades.length) * 100
    : 0;

  return (
    <div className="p-4 max-w-[1200px] mx-auto animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div>
          <h1 className="text-xl font-bold text-[#F1F5F9]">Trades</h1>
          <p className="text-xs text-[#64748B] mt-0.5">{trades.length} trades loaded</p>
        </div>
        <div className="flex items-center gap-2">
          <label
            className={`flex items-center gap-2 px-3 py-2 rounded-lg border border-[#2A3050] text-[#94A3B8] text-sm font-medium hover:text-[#F1F5F9] hover:border-[#3A4060] transition-all ${importing ? "opacity-60 pointer-events-none" : "cursor-pointer"}`}
            title="Import your full trade history from a Groww F&O P&L statement (.xlsx / .csv)"
          >
            <span>⬆</span>
            {importing ? "Importing…" : "Import Groww P&L"}
            <input type="file" accept=".xlsx,.xls,.csv" className="hidden" onChange={handleImport} disabled={importing} />
          </label>
          <button
            onClick={() => setEntryOpen(true)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg ai-gradient text-white text-sm font-medium hover:opacity-90 active:scale-[0.98] transition-all shadow-glow-purple"
          >
            <span>+</span>
            Log Trade
          </button>
        </div>
      </div>
      {importMsg && (
        <div className="mb-3 text-xs px-3 py-2 rounded-lg bg-[#1A2035] border border-[#2A3050] text-[#94A3B8]">
          {importMsg}
        </div>
      )}

      {/* Summary */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="card p-3 text-center">
          <div className="text-[10px] text-[#64748B] uppercase tracking-wide mb-1">Total P&L</div>
          <div
            className="text-xl font-bold font-num"
            style={{ color: totalPnl >= 0 ? "#10B981" : "#EF4444" }}
          >
            {totalPnl >= 0 ? "+" : ""}₹{Math.abs(totalPnl).toLocaleString()}
          </div>
        </div>
        <div className="card p-3 text-center">
          <div className="text-[10px] text-[#64748B] uppercase tracking-wide mb-1">Win Rate</div>
          <div
            className="text-xl font-bold font-num"
            style={{ color: winRate >= 55 ? "#10B981" : winRate >= 40 ? "#F59E0B" : "#EF4444" }}
          >
            {winRate.toFixed(0)}%
          </div>
        </div>
        <div className="card p-3 text-center">
          <div className="text-[10px] text-[#64748B] uppercase tracking-wide mb-1">Total Trades</div>
          <div className="text-xl font-bold font-num text-[#F1F5F9]">{trades.length}</div>
        </div>
      </div>

      {/* Filter */}
      <div className="flex gap-1 mb-4">
        {(["all", "open", "closed"] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilterStatus(f)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              filterStatus === f
                ? "bg-[#8B5CF6]/20 text-[#A78BFA] border border-[#8B5CF6]/30"
                : "text-[#64748B] hover:text-[#94A3B8] bg-[#1A2035]"
            }`}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
            {f === "open" && trades.filter((t) => t.status === "open").length > 0 && (
              <span className="ml-1.5 px-1 rounded-full bg-[#10B981]/20 text-[#10B981] text-[9px]">
                {trades.filter((t) => t.status === "open").length}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Trades Table */}
      <div className="card p-4">
        {/* Header row */}
        <div className="flex items-center gap-3 pb-2 mb-1 border-b border-[#2A3050] text-[10px] uppercase tracking-widest text-[#64748B]">
          <span className="w-12">Dir</span>
          <span className="flex-1">Symbol</span>
          <span className="hidden md:block w-20 text-right">Entry/Exit</span>
          <span className="w-14 text-right">R</span>
          <span className="w-20 text-right">P&L</span>
          <span className="hidden lg:block w-16">Discipline</span>
          <span className="w-20">Status</span>
        </div>

        {loading ? (
          <div className="py-12 text-center text-[#64748B] text-sm">Loading trades...</div>
        ) : trades.length === 0 ? (
          <div className="py-12 text-center">
            <div className="text-3xl mb-3 opacity-30">⟡</div>
            <div className="text-sm text-[#64748B]">No trades found</div>
            <button
              onClick={() => setEntryOpen(true)}
              className="mt-3 text-xs text-[#8B5CF6] hover:text-[#A78BFA]"
            >
              Log your first trade →
            </button>
          </div>
        ) : (
          trades.map((t) => (
            <TradeRow key={t.id} trade={t} onClose={setCloseTradeId} />
          ))
        )}
      </div>

      <TradeEntryDialog isOpen={entryOpen} onClose={() => { setEntryOpen(false); fetchTrades(); }} />
      {closeTradeId && (
        <TradeCloseDialog
          tradeId={closeTradeId}
          isOpen={!!closeTradeId}
          onClose={() => { setCloseTradeId(null); fetchTrades(); }}
        />
      )}
    </div>
  );
}
