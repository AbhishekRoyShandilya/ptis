// src/pages/StrategiesHubPage.tsx
import TabBar, { useTabFromUrl, type TabDef } from "@/components/common/TabBar";
import StrategiesPage from "./StrategiesPage";
import TeachChat from "./TeachChat";
import PatternDiscoveryPage from "./PatternDiscoveryPage";
import TemplatesPage from "./TemplatesPage";

const TABS: TabDef[] = [
  { key: "my", label: "My Strategies", icon: "⟐" },
  { key: "templates", label: "Templates", icon: "◎" },
  { key: "teach", label: "Teach a Strategy", icon: "⚡" },
  { key: "discovery", label: "Discovery", icon: "🔬" },
];

export default function StrategiesHubPage() {
  const [tab, setTab] = useTabFromUrl(TABS);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-[#F1F5F9] mb-4">Strategies</h1>
      <TabBar tabs={TABS} activeTab={tab} onTabChange={setTab} />
      {tab === "my" && <StrategiesPage />}
      {tab === "templates" && <TemplatesPage />}
      {tab === "teach" && <TeachChat />}
      {tab === "discovery" && <PatternDiscoveryPage />}
    </div>
  );
}
