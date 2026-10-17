// src/pages/DashboardPage.tsx
// ONE conversational partner — the chat IS the surface. The old Partner tab is merged in (selecting a
// symbol now makes the partner SPEAK its eye read into the chat, see coPlannerStore.setActiveSymbol), and
// the GEN-1 "Alerts" tab (AlertsPanel polled /cognitive-alerts on realtime_service) is retired — proactive
// alerts now arrive IN the chat (COGNITIVE_ALERT → /ws/session → store) and on the Intelligence hub's
// Live Alerts. Left: Chat (the partner) + Regime (strategy×regime matrix). Feels like a partner, not a
// scanner with modes. (PartnerPanel/AlertsPanel left in the tree, unrouted, pending hard-delete.)
import { useState } from "react";
import WatchlistPanel from "@/components/dashboard/WatchlistPanel";
import ChatPanel from "@/components/dashboard/ChatPanel";
import RegimeCard from "@/components/dashboard/RegimeCard";
import { RegimeMatrixTab } from "@/pages/CoPlannerPage";

type Tab = "chat" | "regime";

const TAB_META: { id: Tab; label: string; icon: string }[] = [
  { id: "chat", label: "Chat", icon: "✦" },
  { id: "regime", label: "Regime", icon: "◆" },
];

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<Tab>("chat");

  return (
    <div className="flex flex-col h-[calc(100vh-56px)] animate-fade-in">
      {/* Market regime strip */}
      <RegimeCard />

      <div className="flex flex-1 min-h-0">
        {/* Left — Watchlist */}
        <div className="w-1/4 min-w-[280px] max-w-[360px] border-r border-[#2A3050] flex flex-col shrink-0">
          <WatchlistPanel />
        </div>

        {/* Right — one flat tabbed panel */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex items-center gap-0.5 px-4 pt-2 pb-0 bg-[#0A0E17] border-b border-[#2A3050] shrink-0">
            {TAB_META.map((t) => (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`relative px-4 py-2 text-xs font-medium rounded-t-lg transition-all ${
                  activeTab === t.id
                    ? "bg-[#0A0E17] text-[#A78BFA] border border-b-0 border-[#2A3050] -mb-px z-10"
                    : "text-[#64748B] hover:text-[#94A3B8] hover:bg-[#1A2035]/50"
                }`}
              >
                <span className="mr-1.5 opacity-70">{t.icon}</span>
                {t.label}
              </button>
            ))}
          </div>

          {/* Content — chat manages its own scroll; regime gets a wrapper */}
          <div className="flex-1 min-h-0">
            {activeTab === "chat" && <ChatPanel />}
            {activeTab === "regime" && (
              <div className="h-full overflow-y-auto p-4">
                <RegimeMatrixTab />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
