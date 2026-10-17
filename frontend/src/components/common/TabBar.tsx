// src/components/common/TabBar.tsx
import { useSearchParams } from "react-router-dom";

export interface TabDef {
  key: string;
  label: string;
  icon?: string;
}

interface TabBarProps {
  tabs: TabDef[];
  activeTab: string;
  onTabChange: (key: string) => void;
}

export function useTabFromUrl(tabs: TabDef[]): [string, (key: string) => void] {
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") || tabs[0]?.key || "";
  const valid = tabs.some((t) => t.key === tab) ? tab : tabs[0]?.key || "";

  const setTab = (key: string) => {
    setParams({ tab: key }, { replace: true });
  };

  return [valid, setTab];
}

export default function TabBar({ tabs, activeTab, onTabChange }: TabBarProps) {
  return (
    <div className="flex items-center gap-1 bg-[#0F1629] border border-[#2A3050] rounded-xl p-1 mb-6">
      {tabs.map((tab) => {
        const active = activeTab === tab.key;
        return (
          <button
            key={tab.key}
            onClick={() => onTabChange(tab.key)}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
              active
                ? "bg-[#8B5CF6]/15 text-[#A78BFA] shadow-sm"
                : "text-[#64748B] hover:text-[#94A3B8] hover:bg-[#1A2035]"
            }`}
          >
            {tab.icon && <span className={active ? "text-[#8B5CF6]" : ""}>{tab.icon}</span>}
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
