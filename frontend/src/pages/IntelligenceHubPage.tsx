// src/pages/IntelligenceHubPage.tsx
// NOTE: the GEN-1 "Pre-Trade Intel" tab (IntelligencePage → /evaluate, /scan/watchlist,
// /environment on realtime_service) was retired — single-symbol eval now lives on the brain
// via the Co-Planner. This hub is brain-only: AI Coach (chat) + Live Alerts (proactive partner
// over /ws/session). IntelligencePage.tsx is left in the tree (unrouted) pending hard delete.
import TabBar, { useTabFromUrl, type TabDef } from "@/components/common/TabBar";
import CoachingPage from "./CoachingPage";
import LiveAlertsPage from "./LiveAlertsPage";

const TABS: TabDef[] = [
  { key: "coach", label: "AI Coach", icon: "✦" },
  { key: "alerts", label: "Live Alerts", icon: "🔔" },
];

export default function IntelligenceHubPage() {
  const [tab, setTab] = useTabFromUrl(TABS);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-[#F1F5F9] mb-4">Intelligence</h1>
      <TabBar tabs={TABS} activeTab={tab} onTabChange={setTab} />
      {tab === "coach" && <CoachingPage />}
      {tab === "alerts" && <LiveAlertsPage />}
    </div>
  );
}
