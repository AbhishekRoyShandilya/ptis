import { useEffect, useState } from "react";
import { avtApi, type SleeveSpec } from "@/services/api";

/**
 * Capital sleeves editor (B.1) — the user SETS how their account is split across instruments (e.g. ₹2L futures /
 * ₹50k options / rest equity). Each pool trades concurrently off its own capital; a full futures book never
 * starves options. Applies on the next AVT (re)start; the API refuses edits while AVT is running.
 */

const CARD = "bg-[#1A2035] border border-[#2A3050] rounded-lg p-4";
const INSTRUMENTS = ["future", "equity", "option"] as const;

type Row = SleeveSpec;

const PRESET: Row[] = [
  { name: "futures", capital: 200000, instruments: ["future"], max_positions: 3, size_mode: "lots", size_value: 1 },
  { name: "options", capital: 50000, instruments: ["option"], max_positions: 2, size_mode: "lots", size_value: 1 },
];

export default function CapitalSleevesConfig() {
  const [virtualCapital, setVirtualCapital] = useState<number>(500000);
  const [rows, setRows] = useState<Row[]>([]);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  const load = () => {
    avtApi.getSleeves().then((c) => {
      setVirtualCapital(c.virtual_capital ?? 500000);
      setRows(c.sleeves ?? []);
      setRunning(c.running);
      setLoading(false);
    }).catch(() => setLoading(false));
  };
  useEffect(load, []);

  const totalAllocated = rows.reduce((s, r) => s + (Number(r.capital) || 0), 0);
  const over = totalAllocated > virtualCapital;

  const update = (i: number, patch: Partial<Row>) =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, ...patch } : r)));
  const toggleInstr = (i: number, inst: string) =>
    update(i, {
      instruments: rows[i].instruments.includes(inst)
        ? rows[i].instruments.filter((x) => x !== inst)
        : [...rows[i].instruments, inst],
    });
  const addRow = () =>
    setRows((rs) => [...rs, { name: "", capital: 0, instruments: [], max_positions: 0, size_mode: "risk_pct", size_value: 0 }]);
  const removeRow = (i: number) => setRows((rs) => rs.filter((_, j) => j !== i));

  const save = async () => {
    setErr(null); setMsg(null);
    try {
      const r = await avtApi.setSleeves({ virtual_capital: virtualCapital, sleeves: rows });
      setMsg(r.message);
    } catch (e: any) {
      setErr(e?.message || "Save failed");
    }
  };
  const reset = async () => {
    setErr(null); setMsg(null);
    try { const r = await avtApi.clearSleeves(); setRows([]); setMsg(r.message); }
    catch (e: any) { setErr(e?.message || "Reset failed"); }
  };

  if (loading) return null;

  return (
    <div className={CARD}>
      <button onClick={() => setOpen((o) => !o)} className="w-full flex items-center justify-between">
        <h3 className="text-xs font-semibold text-[#64748B] uppercase tracking-wider">
          Capital sleeves {rows.length > 0 ? `(${rows.length} pools)` : "(shared pool)"}
        </h3>
        <span className="text-[#64748B] text-xs">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="mt-3 space-y-3">
          <p className="text-[11px] text-[#94A3B8]">
            Split your account into independent pools so futures, options and equity run concurrently off separate
            capital — a full futures book never blocks an option. Applies on the next AVT (re)start.
          </p>

          {running && (
            <div className="text-[11px] text-[#F59E0B] bg-[#F59E0B]/10 border border-[#F59E0B]/30 rounded px-2 py-1">
              AVT is running — stop it before changing capital sleeves.
            </div>
          )}

          <div className="flex items-center gap-2 text-xs">
            <span className="text-[#94A3B8]">Total account ₹</span>
            <input type="number" value={virtualCapital} disabled={running}
              onChange={(e) => setVirtualCapital(Number(e.target.value))}
              className="w-32 bg-[#0B0F1E] border border-[#2A3050] rounded px-2 py-1 text-[#F1F5F9] font-num disabled:opacity-50" />
            <span className={`ml-auto ${over ? "text-[#EF4444]" : "text-[#64748B]"}`}>
              allocated ₹{totalAllocated.toLocaleString()} {over ? "(over — kept as margin/buffer)" : ""}
            </span>
          </div>

          <div className="space-y-2">
            {rows.map((r, i) => (
              <div key={i} className="bg-[#141a2e] border border-[#2A3050] rounded p-2 space-y-2">
                {/* line 1: name · capital · instruments · slots · remove */}
                <div className="grid grid-cols-12 gap-2 items-center">
                  <input placeholder="name" value={r.name} disabled={running}
                    onChange={(e) => update(i, { name: e.target.value })}
                    className="col-span-2 bg-[#0B0F1E] border border-[#2A3050] rounded px-2 py-1 text-xs text-[#F1F5F9] disabled:opacity-50" />
                  <input type="number" placeholder="capital ₹" value={r.capital} disabled={running}
                    onChange={(e) => update(i, { capital: Number(e.target.value) })}
                    className="col-span-3 bg-[#0B0F1E] border border-[#2A3050] rounded px-2 py-1 text-xs text-[#F1F5F9] font-num disabled:opacity-50" />
                  <div className="col-span-4 flex gap-1 flex-wrap">
                    {INSTRUMENTS.map((inst) => (
                      <button key={inst} disabled={running} onClick={() => toggleInstr(i, inst)}
                        className={`text-[10px] px-1.5 py-0.5 rounded border disabled:opacity-50 ${
                          r.instruments.includes(inst)
                            ? "bg-[#3B82F6]/20 border-[#3B82F6] text-[#93C5FD]"
                            : "bg-transparent border-[#2A3050] text-[#64748B]"}`}>
                        {inst}
                      </button>
                    ))}
                  </div>
                  <input type="number" placeholder="slots" value={r.max_positions || ""} disabled={running}
                    title="Max concurrent positions in this pool — how many DIFFERENT stocks at once (0 = engine default)"
                    onChange={(e) => update(i, { max_positions: Number(e.target.value) })}
                    className="col-span-2 bg-[#0B0F1E] border border-[#2A3050] rounded px-2 py-1 text-xs text-[#F1F5F9] font-num disabled:opacity-50" />
                  <button disabled={running} onClick={() => removeRow(i)}
                    className="col-span-1 text-[#EF4444] text-xs disabled:opacity-30">✕</button>
                </div>
                {/* line 2: SIZING — lots for F&O, ₹ amount for equity, or risk-% */}
                <div className="flex items-center gap-2 pl-1">
                  <span className="text-[10px] text-[#64748B] uppercase tracking-wide">Size by</span>
                  <select value={r.size_mode} disabled={running}
                    onChange={(e) => update(i, { size_mode: e.target.value as SleeveSpec["size_mode"] })}
                    className="bg-[#0B0F1E] border border-[#2A3050] rounded px-2 py-1 text-xs text-[#F1F5F9] disabled:opacity-50">
                    <option value="lots">Lots (F&amp;O)</option>
                    <option value="amount">₹ amount (equity)</option>
                    <option value="risk_pct">Risk % (auto)</option>
                  </select>
                  {r.size_mode !== "risk_pct" && (
                    <input type="number" value={r.size_value || ""} disabled={running}
                      placeholder={r.size_mode === "lots" ? "lots" : "₹ per trade"}
                      onChange={(e) => update(i, { size_value: Number(e.target.value) })}
                      className="w-28 bg-[#0B0F1E] border border-[#2A3050] rounded px-2 py-1 text-xs text-[#F1F5F9] font-num disabled:opacity-50" />
                  )}
                  <span className="text-[10px] text-[#64748B]">
                    {r.size_mode === "lots" ? `→ ${r.size_value || 0} lot(s) per trade`
                      : r.size_mode === "amount" ? `→ ₹${(r.size_value || 0).toLocaleString()} per trade`
                      : "→ 1% risk per trade (auto shares)"}
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <button disabled={running} onClick={addRow}
              className="text-[11px] px-2 py-1 rounded border border-[#2A3050] text-[#94A3B8] hover:bg-[#2A3050]/40 disabled:opacity-50">
              + Add pool
            </button>
            <button disabled={running} onClick={() => setRows(PRESET)}
              className="text-[11px] px-2 py-1 rounded border border-[#2A3050] text-[#94A3B8] hover:bg-[#2A3050]/40 disabled:opacity-50">
              Use futures/options preset
            </button>
            <div className="ml-auto flex gap-2">
              <button disabled={running} onClick={reset}
                className="text-[11px] px-3 py-1 rounded bg-[#64748B]/20 text-[#94A3B8] hover:bg-[#64748B]/30 disabled:opacity-50">
                Revert to shared pool
              </button>
              <button disabled={running} onClick={save}
                className="text-[11px] px-3 py-1 rounded bg-[#10B981]/20 text-[#10B981] hover:bg-[#10B981]/30 disabled:opacity-50">
                Save split
              </button>
            </div>
          </div>

          {msg && <div className="text-[11px] text-[#10B981]">{msg}</div>}
          {err && <div className="text-[11px] text-[#EF4444]">{err}</div>}
        </div>
      )}
    </div>
  );
}
