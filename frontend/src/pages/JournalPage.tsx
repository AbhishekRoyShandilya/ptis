// src/pages/JournalPage.tsx
import { useCallback, useEffect, useState } from "react";
import { journalApi } from "@/services/api";
import { JournalDialog } from "@/components/journal/JournalDialog";

const MOOD_EMOJI: Record<string, string> = {
  confident: "😊",
  satisfied: "😌",
  frustrated: "😤",
  anxious: "😰",
  excited: "🤩",
  neutral: "😐",
  low: "😔",
};

export default function JournalPage() {
  const [entries, setEntries] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);

  const fetchEntries = useCallback(() => {
    journalApi
      .list({ limit: 30 })
      .then((d: any) => setEntries(Array.isArray(d) ? d : []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchEntries();
  }, [fetchEntries]);

  // Journal logging now lives here (moved off the top navbar). ⌘J / Ctrl+J opens it.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "j") {
        e.preventDefault();
        setDialogOpen(true);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const closeDialog = () => {
    setDialogOpen(false);
    fetchEntries(); // pick up a newly logged entry
  };

  const LogButton = (
    <button
      onClick={() => setDialogOpen(true)}
      className="flex items-center gap-1.5 px-4 py-2 rounded-lg ai-gradient text-white text-sm font-medium hover:opacity-90 active:scale-[0.98] transition-all shadow-glow-purple"
    >
      <span>🎤</span>
      New Entry
      <kbd className="hidden lg:inline px-1 py-0.5 rounded bg-black/20 border border-white/10 text-[10px]">
        ⌘J
      </kbd>
    </button>
  );

  const Header = (
    <div className="flex items-center justify-between mb-5">
      <div>
        <h1 className="text-xl font-bold text-[#F1F5F9]">Journal</h1>
        <p className="text-xs text-[#64748B] mt-0.5">Your trading diary and reflections</p>
      </div>
      {LogButton}
    </div>
  );

  if (loading) {
    return (
      <div className="p-4 max-w-[800px] mx-auto">
        <div className="text-[#64748B] text-sm animate-pulse">Loading journal…</div>
      </div>
    );
  }

  return (
    <div className="p-4 max-w-[800px] mx-auto animate-fade-in">
      {Header}

      {entries.length === 0 ? (
        <div className="card p-8 text-center">
          <div className="text-3xl mb-3">📓</div>
          <h2 className="text-sm font-semibold text-[#F1F5F9] mb-1">No journal entries yet</h2>
          <p className="text-xs text-[#64748B] max-w-md mx-auto leading-relaxed">
            Your journal is empty. Log a pre-market note, a trade reflection, or
            an end-of-day review and it appears here.
          </p>
          <button
            onClick={() => setDialogOpen(true)}
            className="mt-4 text-xs text-[#8B5CF6] hover:text-[#A78BFA]"
          >
            Write your first entry →
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {entries.map((entry) => (
            <div key={entry.id} className="card p-4 hover:border-[#3A4060] transition-colors">
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  {entry.mood && (
                    <span className="text-lg">{MOOD_EMOJI[entry.mood] || "📝"}</span>
                  )}
                  <div>
                    {entry.symbol && (
                      <span className="text-xs font-bold text-[#F1F5F9] mr-2">{entry.symbol}</span>
                    )}
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#8B5CF6]/10 text-[#A78BFA] capitalize">
                      {entry.type || "note"}
                    </span>
                  </div>
                </div>
                <span className="text-[10px] text-[#64748B]">
                  {new Date(entry.created_at).toLocaleString("en-IN", {
                    dateStyle: "short",
                    timeStyle: "short",
                  })}
                </span>
              </div>
              <p className="text-sm text-[#CBD5E1] leading-relaxed">{entry.content}</p>
            </div>
          ))}
        </div>
      )}

      <JournalDialog isOpen={dialogOpen} onClose={closeDialog} />
    </div>
  );
}
