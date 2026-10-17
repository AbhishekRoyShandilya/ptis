// src/pages/SystemHubPage.tsx
import TabBar, { useTabFromUrl, type TabDef } from "@/components/common/TabBar";
import ShadowTradingPage from "./ShadowTradingPage";
import SKEPage from "./SKEPage";

const TABS: TabDef[] = [
  { key: "shadow", label: "Shadow Trading", icon: "◐" },
  { key: "ske", label: "SKE", icon: "🎯" },
];

export default function SystemHubPage() {
  const [tab, setTab] = useTabFromUrl(TABS);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-[#F1F5F9] mb-4">System</h1>
      <TabBar tabs={TABS} activeTab={tab} onTabChange={setTab} />
      {tab === "shadow" && <ShadowTradingPage />}
      {tab === "ske" && <SKEPage />}
    </div>
  );
}
