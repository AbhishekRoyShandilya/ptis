// src/pages/TeachChat.tsx
// Conversational strategy-teaching chat, with a chat-HISTORY sidebar where each strategy is a reopenable
// session (the sidebar IS "My Strategies"). You talk naturally; PTIS routes each message — a DESCRIPTION,
// a trade EXAMPLE (grounded on the eye), or an ASK when unclear (it asks, never guesses). Say "review" →
// it shows the grounded review (signature, gaps, validation with matched examples, refinements). Name it,
// "Teach it" → the whole session persists (signature + conversation + examples + a new version) and shows
// up in the sidebar to reopen and refine later. Organs reason; the LLM only routes/extracts (guarded).
import { useState, useRef, useEffect, useCallback } from "react";
import { intelligenceApi } from "@/services/api";
import AnalysisPanel from "./AnalysisPanel";

interface Grounded { symbol: string; when: string; label: string; ok: boolean; state?: string; stage?: string; tags?: string[]; note?: string }
interface Stats { n?: number; ran_rate?: number; base_ran_rate?: number; lift?: number; validated?: boolean }
interface VerifyEx { symbol: string; date: string; time: string; dir: string; peak_ext: number; outcome: string }
interface Refinement { add: string; lift: number; lift_delta: number; validated: boolean }
interface Review {
  direction: string; require: string[]; exclude: string[]; execution_notes: string[]; gaps: string[];
  validation?: { stats: Stats; verify_examples: VerifyEx[] }; refinements: Refinement[];
  explanation?: { concept: string; your_words: string; logic: { tag: string; means: string }[]; exclude_logic: { tag: string; means: string }[]; evidence: string };
}
interface Msg { id: number; role: "user" | "bot"; text?: string; ground?: Grounded; review?: Review }
interface SessionMeta { name: string; direction: string; require: string[]; exclude: string[]; versions: number; examples: number; updated: string; total_R?: number | null }

const pct = (x?: number) => (x === undefined ? "—" : `${Math.round(x * 100)}%`);
let _id = 0;
const WELCOME: Msg = { id: _id++, role: "bot", text: "Tell me how you trade — describe your strategy in your own words (English / Hindi / Hinglish). Point me to example moments like “HAL 22 April 2026 11:15, that one failed”, and I'll show you what the eye saw. Say “review” when you're ready." };

export default function TeachChat() {
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [activeName, setActiveName] = useState<string>("");
  const [mode, setMode] = useState<"teaching" | "analysis">("teaching");
  const [analysisStrat, setAnalysisStrat] = useState<SessionMeta | null>(null);

  const [msgs, setMsgs] = useState<Msg[]>([WELCOME]);
  const [description, setDescription] = useState("");
  const [examples, setExamples] = useState<{ symbol: string; when: string; label: string }[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [require, setRequire] = useState<string[]>([]);
  const [exclude, setExclude] = useState<string[]>([]);
  const [direction, setDirection] = useState("up");
  const [name, setName] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  const loadSessions = useCallback(async () => {
    try { const r = await intelligenceApi.teachSessions() as { sessions: SessionMeta[] }; setSessions(r.sessions || []); } catch { /* */ }
  }, []);
  useEffect(() => { loadSessions(); }, [loadSessions]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs, busy]);
  const push = (m: Omit<Msg, "id">) => setMsgs((p) => [...p, { ...m, id: _id++ }]);

  function newSession() {
    setMode("teaching"); setAnalysisStrat(null);
    setActiveName(""); setMsgs([{ ...WELCOME, id: _id++ }]); setDescription(""); setExamples([]);
    setRequire([]); setExclude([]); setDirection("up"); setName(""); setInput("");
  }

  function openAnalysis(s: SessionMeta) {
    setMode("analysis"); setAnalysisStrat(s); setActiveName(s.name);
  }

  async function openSession(n: string) {
    setMode("teaching"); setAnalysisStrat(null);
    setBusy(true);
    try {
      const s = await intelligenceApi.teachSession(n) as { name: string; direction: string; require: string[]; exclude: string[]; description?: string; messages?: Msg[]; examples?: { symbol: string; when: string; label: string }[] };
      setActiveName(s.name); setName(s.name); setDirection(s.direction || "up");
      setRequire(s.require || []); setExclude(s.exclude || []); setDescription(s.description || "");
      setExamples(s.examples || []);
      const restored = (s.messages || []).map((m) => ({ ...m, id: _id++ }));
      const base: Msg[] = [
        { ...WELCOME, id: _id++ },
        { id: _id++, role: "bot", text: `Reopened “${s.name}”. Here's what I understand so far — point me to more examples or tell me what to change.` },
        ...restored,
      ];
      // current understanding, reconstructed from the stored signature (NOT a replay of your exact words)
      let review: Review | null = null;
      try { review = await intelligenceApi.teachReviewSignature({ direction: s.direction || "up", require: s.require || [], exclude: s.exclude || [] }) as Review; } catch { /* show conversation even if review fails */ }
      setMsgs(review ? [...base, { id: _id++, role: "bot", review }] : base);
    } catch (e) {
      push({ role: "bot", text: "Couldn't reopen: " + (e instanceof Error ? e.message : "error") });
    } finally { setBusy(false); }
  }

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput(""); push({ role: "user", text }); setBusy(true);
    try {
      if (/^(review|validate|done|that'?s it)\b/i.test(text)) { await runReview(); }
      else {
        const r = await intelligenceApi.teachExtract(text) as { kind: string; ask?: string; symbol?: string; when?: string; label?: string };
        if (r.kind === "question") {
          const a = await intelligenceApi.teachAsk({ question: text, direction, require, exclude }) as { answer: string };
          push({ role: "bot", text: a.answer });
        } else if (r.kind === "ask") push({ role: "bot", text: r.ask });
        else if (r.kind === "example" && r.symbol && r.when) {
          const g = await intelligenceApi.teachGround({ symbol: r.symbol, when: r.when, label: r.label }) as Grounded;
          setExamples((p) => [...p, { symbol: r.symbol!, when: r.when!, label: r.label || "take" }]);
          push({ role: "bot", ground: g });
        } else { setDescription((d) => (d ? d + "\n" : "") + text); push({ role: "bot", text: "Got it — captured. Add more, point me to examples, or say “review”." }); }
      }
    } catch (e) { push({ role: "bot", text: "Something went wrong: " + (e instanceof Error ? e.message : "error") }); }
    finally { setBusy(false); }
  }

  async function runReview() {
    if (!description.trim()) { push({ role: "bot", text: "Describe the strategy first — when do you take it, what do you need to see, what makes you skip it?" }); return; }
    const r = await intelligenceApi.teachBootstrap({ description, examples }) as Review;
    setRequire(r.require); setExclude(r.exclude); setDirection(r.direction);
    push({ role: "bot", review: r });
  }

  async function store() {
    if (!name.trim() || require.length === 0) return;
    setBusy(true);
    try {
      const finalMsgs: Msg[] = [...msgs];  // capture current
      await intelligenceApi.teachSaveSession({ name: name.trim(), direction, require, exclude, description, messages: finalMsgs, examples });
      push({ role: "bot", text: `✓ Stored “${name.trim()}”. It's in the sidebar — reopen it anytime to refine.` });
      setActiveName(name.trim()); await loadSessions();
    } catch (e) { push({ role: "bot", text: "Couldn't store: " + (e instanceof Error ? e.message : "error") }); }
    finally { setBusy(false); }
  }

  return (
    <div className="flex gap-3 h-[calc(100vh-150px)]">
      {/* sidebar = My Strategies / chat history of strategy-sessions */}
      <div className="w-56 shrink-0 border-r border-[#2A3050] pr-2 flex flex-col">
        <button onClick={newSession} className="mb-2 px-3 py-2 rounded-lg text-xs font-medium bg-[#8B5CF6]/15 border border-[#8B5CF6]/30 text-[#A78BFA] hover:bg-[#8B5CF6]/25">+ New strategy</button>
        <div className="text-[10px] uppercase tracking-wide text-[#3A4060] px-1 mb-1">My strategies</div>
        <div className="flex-1 overflow-y-auto space-y-1">
          {sessions.map((s) => {
            const active = activeName === s.name;
            return (
              <div key={s.name} className={`group rounded-lg px-2.5 py-2 transition-colors ${active ? "bg-[#8B5CF6]/15" : "hover:bg-[#1A2035]"}`}>
                <button onClick={() => openSession(s.name)} className="w-full text-left">
                  <div className={`truncate text-xs font-medium ${active ? "text-[#A78BFA]" : "text-[#94A3B8]"}`}>{s.name}</div>
                  <div className="text-[10px] text-[#64748B]">{s.direction} · v{s.versions} · {s.examples} ex{s.total_R !== undefined && s.total_R !== null ? <span style={{ color: s.total_R >= 0 ? "#10B981" : "#EF4444" }}> · {s.total_R >= 0 ? "+" : ""}{s.total_R}R</span> : ""}</div>
                </button>
                {/* two doors */}
                <div className="flex gap-1 mt-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button onClick={() => openSession(s.name)} className={`flex-1 px-1.5 py-1 rounded text-[10px] border ${active && mode === "teaching" ? "border-[#8B5CF6]/50 text-[#A78BFA] bg-[#8B5CF6]/10" : "border-[#2A3050] text-[#64748B] hover:text-[#A78BFA]"}`}>Teaching</button>
                  <button onClick={() => openAnalysis(s)} className={`flex-1 px-1.5 py-1 rounded text-[10px] border ${active && mode === "analysis" ? "border-[#8B5CF6]/50 text-[#A78BFA] bg-[#8B5CF6]/10" : "border-[#2A3050] text-[#64748B] hover:text-[#A78BFA]"}`}>Analysis</button>
                </div>
              </div>
            );
          })}
          {sessions.length === 0 && <div className="text-[11px] text-[#64748B] px-1">None yet — teach your first.</div>}
        </div>
      </div>

      {/* right panel: teaching chat OR analysis (money) */}
      {mode === "analysis" && analysisStrat ? (
        <AnalysisPanel strat={analysisStrat} />
      ) : (
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 overflow-y-auto px-1 py-3 space-y-3">
            {msgs.map((m) => <Bubble key={m.id} m={m} require={require} exclude={exclude} direction={direction}
              onRefine={(t) => setRequire((p) => p.includes(t) ? p : [...p, t])} name={name} setName={setName} onStore={store} busy={busy} />)}
            {busy && <div className="text-xs text-[#64748B] px-2">thinking…</div>}
            <div ref={endRef} />
          </div>
          <div className="flex items-end gap-2 border-t border-[#2A3050] pt-3 px-1">
            <textarea value={input} onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              rows={1} placeholder="Describe your strategy, or point to an example (e.g. HAL 22 Apr 2026 11:15, it failed)…"
              className="flex-1 bg-[#1A2035] border border-[#2A3050] rounded-xl px-3 py-2.5 text-sm text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none focus:border-[#8B5CF6]/50 resize-none" />
            <button onClick={runReview} disabled={busy} className="px-3 py-2 rounded-lg text-xs bg-[#1A2035] border border-[#2A3050] text-[#94A3B8] hover:text-[#A78BFA]">Review</button>
            <button onClick={send} disabled={busy || !input.trim()} className="px-4 py-2 rounded-lg text-sm font-medium bg-[#8B5CF6] text-white hover:bg-[#7C3AED] disabled:opacity-40">Send</button>
          </div>
        </div>
      )}
    </div>
  );
}

function Chip({ text, tone = "req" }: { text: string; tone?: "req" | "exc" | "gap" }) {
  const c = tone === "exc" ? "#EF4444" : tone === "gap" ? "#F59E0B" : "#8B5CF6";
  return <span className="px-2 py-0.5 rounded text-[11px] font-mono border" style={{ color: c, borderColor: c + "55", background: c + "12" }}>{text}</span>;
}

function Bubble({ m, require, exclude, direction, onRefine, name, setName, onStore, busy }: {
  m: Msg; require: string[]; exclude: string[]; direction: string; onRefine: (t: string) => void;
  name: string; setName: (s: string) => void; onStore: () => void; busy: boolean;
}) {
  if (m.role === "user") return <div className="flex justify-end"><div className="max-w-[80%] px-3 py-2 rounded-2xl rounded-tr-sm bg-[#8B5CF6]/20 border border-[#8B5CF6]/30 text-sm text-[#F1F5F9] whitespace-pre-wrap">{m.text}</div></div>;

  if (m.ground) {
    const g = m.ground;
    return (
      <div className="flex justify-start"><div className="max-w-[88%] px-3 py-2 rounded-2xl rounded-tl-sm bg-[#11162A] border border-[#2A3050] text-[12px]">
        <span className="font-mono text-[#F1F5F9]">{g.symbol} {g.when}</span> · <span className={g.label === "take" ? "text-[#10B981]" : g.label === "fail" ? "text-[#EF4444]" : "text-[#94A3B8]"}>{g.label}</span>
        {g.ok ? <div className="mt-1 text-[#94A3B8]">the eye saw <span className="text-[#CBD5E1]">{g.state}{g.stage ? ` · ${g.stage}` : ""}</span><div className="mt-1 flex flex-wrap gap-1">{(g.tags || []).map((t) => <Chip key={t} text={t} />)}</div></div>
                : <div className="mt-1 text-[#F59E0B]">{g.note}</div>}
      </div></div>
    );
  }

  if (m.review) {
    const r = m.review; const st = r.validation?.stats;
    return (
      <div className="flex justify-start"><div className="w-full px-3.5 py-3 rounded-2xl rounded-tl-sm bg-[#1A2035] border border-[#2A3050] space-y-3">
        <div className="text-xs font-semibold text-[#A78BFA]">Here's what I understood — verify it</div>
        {r.explanation && (
          <div className="px-3 py-2.5 rounded-lg bg-[#11162A] border border-[#2A3050] space-y-2">
            <div className="text-[13px] text-[#E2E8F0] leading-relaxed">{r.explanation.concept}</div>
            {r.explanation.your_words && <div className="text-[11px] text-[#94A3B8] italic border-l-2 border-[#2A3050] pl-2">“{r.explanation.your_words}”</div>}
            {(r.explanation.logic.length > 0 || r.explanation.exclude_logic.length > 0) && (
              <div className="pt-0.5 space-y-1">
                <div className="text-[10px] uppercase tracking-wide text-[#3A4060]">the logic — what must be true</div>
                {r.explanation.logic.map((x) => <div key={x.tag} className="text-[11px] flex gap-2"><span className="font-mono text-[#8B5CF6] w-32 shrink-0">{x.tag}</span><span className="text-[#94A3B8]">{x.means}</span></div>)}
                {r.explanation.exclude_logic.map((x) => <div key={x.tag} className="text-[11px] flex gap-2"><span className="font-mono text-[#EF4444] w-32 shrink-0">avoid · {x.tag}</span><span className="text-[#94A3B8]">{x.means}</span></div>)}
              </div>
            )}
            {r.explanation.evidence && <div className="text-[11px] text-[#10B981] pt-0.5">{r.explanation.evidence}</div>}
          </div>
        )}
        <div className="flex flex-wrap items-center gap-1.5 text-xs"><span className="text-[#64748B]">signature</span><Chip text={direction} /><span className="text-[#64748B] ml-1">require</span>{require.map((t) => <Chip key={t} text={t} />)}{exclude.map((t) => <Chip key={t} text={t} tone="exc" />)}</div>
        {r.gaps.length > 0 && <div className="px-2.5 py-2 rounded-lg border border-[#F59E0B]/30 bg-[#F59E0B]/5"><div className="text-[11px] font-semibold text-[#F59E0B] mb-1">I can't see these yet</div><div className="flex flex-wrap gap-1.5">{r.gaps.map((g, i) => <Chip key={i} text={g} tone="gap" />)}</div></div>}
        {r.execution_notes.length > 0 && <div className="text-[11px] text-[#94A3B8]"><span className="text-[#64748B]">you operate: </span>{r.execution_notes.join(" · ")}</div>}
        {st && (
          <div className="px-2.5 py-2 rounded-lg bg-[#11162A] border border-[#2A3050]">
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="text-[#94A3B8]">n={st.n}</span><span className="text-[#CBD5E1]">{pct(st.ran_rate)} <span className="text-[#64748B]">vs {pct(st.base_ran_rate)} base</span></span>
              <span style={{ color: (st.lift ?? 0) > 0 ? "#10B981" : "#EF4444" }}>{(st.lift ?? 0) > 0 ? "+" : ""}{Math.round((st.lift ?? 0) * 100)} lift</span>
              {st.validated ? <span className="px-1.5 py-0.5 rounded bg-[#10B981]/15 text-[#10B981] font-semibold">✓ VALIDATED</span> : <span className="px-1.5 py-0.5 rounded bg-[#64748B]/15 text-[#94A3B8]">not validated</span>}
              <span className="text-[#64748B] text-[10px]">(11-symbol library — provisional)</span>
            </div>
            {r.validation && r.validation.verify_examples.length > 0 && (
              <div className="mt-2"><div className="text-[10px] text-[#64748B] mb-1">are these your setup?</div>
                {r.validation.verify_examples.slice(0, 5).map((e, i) => <div key={i} className="text-[11px] font-mono text-[#94A3B8]">{e.symbol} {e.date} {e.dir} +{e.peak_ext} ATR → <span style={{ color: e.outcome === "RAN" ? "#10B981" : e.outcome === "FAILED" ? "#EF4444" : "#94A3B8" }}>{e.outcome}</span></div>)}
              </div>
            )}
          </div>
        )}
        {r.refinements.length > 0 && (
          <div><div className="text-[11px] text-[#94A3B8] mb-1">refinements (click to add):</div><div className="flex flex-wrap gap-1.5">
            {r.refinements.slice(0, 5).map((x) => <button key={x.add} onClick={() => onRefine(x.add)} disabled={require.includes(x.add)} className="px-2 py-1 rounded text-[11px] font-mono border border-[#10B981]/40 bg-[#10B981]/10 text-[#10B981] hover:bg-[#10B981]/20 disabled:opacity-40">+{x.add} <span className="text-[#64748B]">(Δ{x.lift_delta > 0 ? "+" : ""}{Math.round(x.lift_delta * 100)})</span></button>)}
          </div></div>
        )}
        <div className="flex items-center gap-2 border-t border-[#2A3050] pt-2.5">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name this strategy" className="flex-1 bg-[#11162A] border border-[#2A3050] rounded px-3 py-1.5 text-sm text-[#F1F5F9] placeholder-[#3A4060] focus:outline-none" />
          <button onClick={onStore} disabled={busy || !name.trim() || require.length === 0} className="px-3 py-1.5 rounded-lg text-sm font-medium bg-[#10B981] text-white hover:bg-[#0EA371] disabled:opacity-40">Teach it</button>
        </div>
      </div></div>
    );
  }

  return <div className="flex justify-start"><div className="max-w-[88%] px-3 py-2 rounded-2xl rounded-tl-sm bg-[#11162A] border border-[#2A3050] text-sm text-[#CBD5E1] whitespace-pre-wrap">{m.text}</div></div>;
}
