// src/components/layout/MainLayout.tsx
import { useState, useEffect } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { TopBar } from "./TopBar";
import { FloatingTradeButton } from "@/components/trade/FloatingTradeButton";
import { TradeEntryDialog } from "@/components/trade/TradeEntryDialog";
import { AlertQueue } from "@/components/common/AlertQueue";
import { behaviorApi } from "@/services/api";

export default function MainLayout() {
  const location = useLocation();
  const isDashboard = location.pathname === "/";
  const [tradeOpen, setTradeOpen] = useState(false);
  const [disciplineScore, setDisciplineScore] = useState<number | null>(null);

  useEffect(() => {
    behaviorApi.disciplineScore()
      .then((d: any) => setDisciplineScore(d?.score ?? null))
      .catch(() => {});
  }, []);

  // Keyboard shortcuts (journal logging now lives in the Journal tab)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "t") {
        e.preventDefault();
        setTradeOpen(true);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <div className="min-h-screen bg-[#0A0E17] flex flex-col">
      <TopBar disciplineScore={disciplineScore} />

      {/* Alert Queue (behavioral alerts) */}
      <AlertQueue />

      {/* Page content */}
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>

      {/* Floating quick-trade button (hidden on dashboard to avoid overlapping chat input) */}
      {!isDashboard && <FloatingTradeButton onClick={() => setTradeOpen(true)} />}

      {/* Dialogs */}
      <TradeEntryDialog
        isOpen={tradeOpen}
        onClose={() => setTradeOpen(false)}
      />
    </div>
  );
}
