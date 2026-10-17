// src/pages/TemplatesPage.tsx — the system's OWN strategy templates (cascade futures/options). The user
// SELECTS one (can't teach it); activating auto-wires its SKOs into their AVT on (re)start.
import { useEffect, useState } from "react";
import { strategiesApi } from "@/services/api";

type Template = {
  id: string;
  name: string;
  one_line: string;
  assets: string;
  instrument: string;
  activated: boolean;
  teachable: boolean;
};

export default function TemplatesPage() {
  const [items, setItems] = useState<Template[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState<string>("");

  const load = async () => {
    setLoading(true);
    try {
      const r: any = await strategiesApi.templates();
      setItems(r?.templates || []);
    } catch {
      setItems([]);
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    load();
  }, []);

  const toggle = async (t: Template) => {
    setBusy(t.id);
    try {
      const r: any = t.activated
        ? await strategiesApi.deactivateTemplate(t.id)
        : await strategiesApi.activateTemplate(t.id);
      setMsg(r?.message || "");
      await load();
    } catch (e: any) {
      setMsg(e?.message || "Failed — is your backend running?");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div>
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-[#F1F5F9]">Strategy Templates</h2>
        <p className="text-sm text-slate-400">
          The system&apos;s own validated strategies. You don&apos;t teach these — you activate one and it
          auto-wires into your AVT, which scans, ranks and takes virtual trades on it.
        </p>
      </div>

      {msg && <div className="mb-3 text-xs text-emerald-400">{msg}</div>}

      {loading ? (
        <div className="text-slate-500">Loading templates…</div>
      ) : items.length === 0 ? (
        <div className="text-slate-500">No templates available.</div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {items.map((t) => (
            <div
              key={t.id}
              className={`rounded-xl border p-5 transition-colors ${
                t.activated ? "border-indigo-500/60 bg-indigo-500/5" : "border-slate-700/60 bg-slate-800/30"
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-[#F1F5F9]">{t.name}</span>
                    <span className="rounded px-2 py-0.5 text-[10px] uppercase tracking-wide bg-slate-700/60 text-slate-300">
                      {t.instrument}
                    </span>
                    {t.activated && (
                      <span className="rounded px-2 py-0.5 text-[10px] uppercase tracking-wide bg-emerald-500/20 text-emerald-300">
                        active
                      </span>
                    )}
                  </div>
                  <div className="mt-1 text-xs text-slate-400">{t.assets}</div>
                </div>
                <button
                  disabled={busy === t.id}
                  onClick={() => toggle(t)}
                  className={`shrink-0 rounded-lg px-3 py-1.5 text-sm font-medium disabled:opacity-50 ${
                    t.activated
                      ? "bg-slate-700 text-slate-200 hover:bg-slate-600"
                      : "bg-indigo-600 text-white hover:bg-indigo-500"
                  }`}
                >
                  {busy === t.id ? "…" : t.activated ? "Deactivate" : "Activate"}
                </button>
              </div>
              <p className="mt-3 text-sm text-slate-300">{t.one_line}</p>
            </div>
          ))}
        </div>
      )}

      <p className="mt-4 text-xs text-slate-500">
        After activating, (re)start your AVT (AVT tab → Start) so it loads the template and begins scanning +
        taking virtual trades on it — with real F&amp;O charges applied.
      </p>
    </div>
  );
}
