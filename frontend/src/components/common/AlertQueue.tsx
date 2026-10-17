// src/components/common/AlertQueue.tsx
import { useState, useEffect } from "react";
import { behaviorApi } from "@/services/api";

interface Alert {
  id: string;
  // type may be one of the known values or other strings produced by backend
  type: "critical" | "warning" | "nudge" | string;
  message: string;
  action?: string;
}

export function AlertQueue() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());

  useEffect(() => {
    // Fetch behavioral deviations as alerts
    behaviorApi.deviations()
      .then((data: any) => {
        if (Array.isArray(data)) {
          const mapped = data.slice(0, 2).map((d: any, i: number) => ({
            id: d.id || String(i),
            type: d.severity === "high" ? "critical" : d.severity === "medium" ? "warning" : "nudge",
            message: d.description || d.message || "Behavioral pattern detected",
            action: d.action,
          }));
          setAlerts(mapped);
        }
      })
      .catch(() => {});
  }, []);

  const visible = alerts.filter((a) => !dismissed.has(a.id));

  if (visible.length === 0) return null;

  return (
    <div className="px-4 py-2 space-y-1.5 border-b border-[#2A3050] bg-[#0A0E17]">
      {visible.map((alert) => {
        const colors = {
          critical: { border: "#EF4444", bg: "rgba(239,68,68,0.06)", icon: "🚨", text: "#EF4444" },
          warning: { border: "#F59E0B", bg: "rgba(245,158,11,0.06)", icon: "⚠️", text: "#F59E0B" },
          nudge: { border: "#3B82F6", bg: "rgba(59,130,246,0.06)", icon: "💡", text: "#3B82F6" },
        }[alert.type] || { border: "#888", bg: "transparent", icon: "", text: "#888" };

        return (
          <div
            key={alert.id}
            className="flex items-center gap-2 px-3 py-2 rounded-lg border text-xs animate-slide-up"
            style={{ borderColor: colors.border + "30", background: colors.bg }}
          >
            <span>{colors.icon}</span>
            <span className="flex-1 text-[#CBD5E1]">{alert.message}</span>
            {alert.action && (
              <span className="text-[11px] px-2 py-0.5 rounded" style={{ color: colors.text, background: colors.border + "15" }}>
                {alert.action}
              </span>
            )}
            <button
              onClick={() => setDismissed((s) => new Set([...s, alert.id]))}
              className="text-[#3A4060] hover:text-[#64748B] text-base leading-none ml-1"
            >
              ×
            </button>
          </div>
        );
      })}
    </div>
  );
}
