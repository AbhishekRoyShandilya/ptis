// src/components/dashboard/CoPlanPanel.tsx
// The Dashboard's single co-plan surface, organised by JOB (not by engine).
// PARTNER is the per-symbol front door — it already speaks the coach voice + stance + action +
// strategy match, so the old "Coach" and "Quick Eval" tabs were redundant skins of the same brain
// read and were collapsed into it (their components remain in the codebase, just not surfaced).
import { useState } from "react";
import ChatPanel from "./ChatPanel";
import { PlanningTab, RegimeMatrixTab } from "@/pages/CoPlannerPage";

// PARTNER is no longer a separate tab — the chat IS the partner: it grounds every symbol read in the eye
// and the partner now SPEAKS into the chat when you pick a symbol (see coPlannerStore.setActiveSymbol).
// One conversational surface → feels like a trading partner, not a scanner with modes.
type Sub = "chat" | "plan" | "matrix";

const SUBS: { id: Sub; label: string }[] = [
  { id: "chat", label: "Chat" },
  { id: "plan", label: "Planning" },
  { id: "matrix", label: "Regime Matrix" },
];

export default function CoPlanPanel() {
  const [sub, setSub] = useState<Sub>("chat");

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Sub-tab bar */}
      <div className="flex items-center gap-1 px-4 py-1.5 border-b border-[#2A3050] shrink-0">
        {SUBS.map((s) => (
          <button
            key={s.id}
            onClick={() => setSub(s.id)}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
              sub === s.id
                ? "bg-[#8B5CF6]/15 text-[#A78BFA]"
                : "text-[#64748B] hover:text-[#94A3B8] hover:bg-[#1A2035]"
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>

      {/* Content — chat & partner manage their own scroll; engine tabs get a scroll wrapper */}
      <div className="flex-1 min-h-0">
        {sub === "chat" ? (
          <ChatPanel />
        ) : (
          <div className="h-full overflow-y-auto p-4">
            {sub === "plan" && <PlanningTab />}
            {sub === "matrix" && <RegimeMatrixTab />}
          </div>
        )}
      </div>
    </div>
  );
}
