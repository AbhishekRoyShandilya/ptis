// src/components/common/LiveAlertList.tsx
import React from "react";

export interface AlertItem {
  id: string;
  priority: string;
  message: string;
  timestamp: string;
}

interface Props {
  alerts: AlertItem[];
}

export function LiveAlertList({ alerts }: Props) {
  if (alerts.length === 0) return <p className="text-gray-400">No alerts.</p>;

  return (
    <ul className="space-y-2">
      {alerts.map((a) => (
        <li key={a.id} className="p-3 border rounded bg-[#1A1E2E]">
          <div className="text-sm text-[#CBD5E1]">
            <span className="font-medium">{a.priority}</span> {a.message}
          </div>
          <div className="text-[10px] text-[#64748B] mt-1">{a.timestamp}</div>
        </li>
      ))}
    </ul>
  );
}
