// src/pages/AnalyticsHubPage.tsx
import TabBar, { useTabFromUrl, type TabDef } from "@/components/common/TabBar";
import AnalyticsPage from "./AnalyticsPage";
import BehaviorPage from "./BehaviorPage";
import ProfilePage from "./ProfilePage";
import RegimePage from "./RegimePage";

const TABS: TabDef[] = [
  { key: "performance", label: "Performance", icon: "◈" },
  { key: "behavior", label: "Behavior", icon: "◎" },
  { key: "profile", label: "Profile", icon: "👤" },
  { key: "regime", label: "Regime", icon: "◉" },
];

export default function AnalyticsHubPage() {
  const [tab, setTab] = useTabFromUrl(TABS);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-[#F1F5F9] mb-4">Analytics</h1>
      <TabBar tabs={TABS} activeTab={tab} onTabChange={setTab} />
      {tab === "performance" && <AnalyticsPage />}
      {tab === "behavior" && <BehaviorPage />}
      {tab === "profile" && <ProfilePage />}
      {tab === "regime" && <RegimePage />}
    </div>
  );
}
