// src/pages/LiveAlertsPage.tsx
import { useEffect, useState, useRef, useCallback } from "react";

// ── Types ─────────────────────────────────────────────────────────

interface CognitiveAlert {
  id: string;
  type: string;
  severity: "critical" | "warning" | "info";
  title: string;
  message: string;
  action_required?: string;
  trade_id?: string;
  symbol?: string;
  metadata?: Record<string, any>;
  timestamp: string;
  acknowledged: boolean;
}

// ── Constants ─────────────────────────────────────────────────────

const WS_URL = import.meta.env.VITE_API_URL
  ? import.meta.env.VITE_API_URL.replace(/^http/, "ws") + "/api/v1/ws/session"
  : `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/api/v1/ws/session`;

const SEVERITY_STYLE = {
  critical: { text: "#EF4444", bg: "#EF444412", border: "#EF444440" },
  warning:  { text: "#F59E0B", bg: "#F59E0B12", border: "#F59E0B40" },
  info:     { text: "#3B82F6", bg: "#3B82F612", border: "#3B82F640" },
} as const;

const TYPE_ICON: Record<string, string> = {
  stop_widening:     "↔",
  risk_deviation:    "⚖",
  emotional_override:"⚡",
  regime_mismatch:   "⊗",
  volatility_spike:  "〜",
  daily_loss_limit:  "⛔",
  overtrading:       "↻",
  portfolio_heat:    "◎",
  drawdown_warning:  "↓",
  discipline_breach: "◈",
  override:          "!",
};

const TYPE_LABEL: Record<string, string> = {
  stop_widening:     "Stop Widening",
  risk_deviation:    "Risk Deviation",
  emotional_override:"Emotional Override",
  regime_mismatch:   "Regime Mismatch",
  volatility_spike:  "Volatility Spike",
  daily_loss_limit:  "Daily Loss Limit",
  overtrading:       "Overtrading",
  portfolio_heat:    "Portfolio Heat",
  drawdown_warning:  "Drawdown Warning",
  discipline_breach: "Discipline Breach",
};

// ── Alert card ────────────────────────────────────────────────────

function AlertCard({
  alert,
  onAcknowledge,
}: {
  alert: CognitiveAlert;
  onAcknowledge: (id: string) => void;
}) {
  const style = SEVERITY_STYLE[alert.severity] ?? SEVERITY_STYLE.info;
  const icon  = TYPE_ICON[alert.type]   ?? "!";

  return (
    <div
      className={`p-4 rounded-xl border transition-all duration-300 ${
        alert.acknowledged ? "opacity-35" : ""
      }`}
      style={{ borderColor: style.border, background: style.bg }}
    >
      <div className="flex items-start gap-3">
        {/* Severity icon */}
        <div
          className="h-9 w-9 rounded-xl flex items-center justify-center shrink-0 text-base font-bold"
          style={{ background: style.text + "20", color: style.text }}
        >
          {icon}
        </div>

        {/* Body */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div>
              <div className="text-sm font-semibold text-[#F1F5F9] leading-tight">
                {alert.title}
              </div>
              {alert.symbol && (
                <div className="text-[10px] text-[#64748B] mt-0.5">{alert.symbol}</div>
              )}
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <span
                className="text-[9px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wide"
                style={{ background: style.text + "20", color: style.text }}
              >
                {alert.severity}
              </span>
              {!alert.acknowledged && (
                <button
                  onClick={() => onAcknowledge(alert.id)}
                  title="Acknowledge"
                  className="text-[#3A4060] hover:text-[#94A3B8] transition-colors text-sm leading-none"
                >
                  ✕
                </button>
              )}
            </div>
          </div>

          <div className="text-xs text-[#94A3B8] mt-1.5 leading-relaxed">{alert.message}</div>

          {alert.action_required && !alert.acknowledged && (
            <div
              className="mt-2 text-[11px] font-medium px-3 py-2 rounded-lg leading-relaxed"
              style={{ background: style.text + "12", color: style.text }}
            >
              ↳ {alert.action_required}
            </div>
          )}

          {alert.metadata && Object.keys(alert.metadata).length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {Object.entries(alert.metadata).map(([k, v]) => (
                <span
                  key={k}
                  className="text-[9px] px-1.5 py-0.5 rounded bg-[#0F1420] border border-[#2A3050]"
                >
                  <span className="text-[#64748B]">{k.replace(/_/g, " ")}: </span>
                  <span className="text-[#94A3B8]">{String(v)}</span>
                </span>
              ))}
            </div>
          )}

          <div className="text-[9px] text-[#3A4060] mt-2">
            {new Date(alert.timestamp).toLocaleTimeString("en-IN", {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            })}
            {alert.trade_id && (
              <span className="ml-2 text-[#3A4060]">Trade: {alert.trade_id.slice(0, 8)}…</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Filter bar ────────────────────────────────────────────────────

type SeverityFilter = "all" | "critical" | "warning" | "info";

function FilterBar({
  filter,
  setFilter,
  showAck,
  setShowAck,
}: {
  filter: SeverityFilter;
  setFilter: (f: SeverityFilter) => void;
  showAck: boolean;
  setShowAck: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      {(["all", "critical", "warning", "info"] as SeverityFilter[]).map((f) => {
        const color =
          f === "critical" ? "#EF4444" :
          f === "warning"  ? "#F59E0B" :
          f === "info"     ? "#3B82F6" : "#64748B";
        return (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1 rounded-full text-[11px] font-medium capitalize transition-all ${
              filter === f
                ? "text-white"
                : "border border-[#2A3050] text-[#64748B] hover:border-[#3A4060]"
            }`}
            style={filter === f ? { background: color } : {}}
          >
            {f}
          </button>
        );
      })}
      <div className="ml-auto flex items-center gap-1.5">
        <span className="text-[11px] text-[#64748B]">Show acked</span>
        <button
          onClick={() => setShowAck(!showAck)}
          className={`relative h-4.5 w-8 rounded-full transition-colors ${
            showAck ? "bg-[#8B5CF6]" : "bg-[#2A3050]"
          }`}
          style={{ height: 18, width: 32 }}
        >
          <span
            className="absolute top-0.5 h-3.5 w-3.5 rounded-full bg-white transition-transform"
            style={{
              height: 14,
              width: 14,
              transform: showAck ? "translateX(15px)" : "translateX(2px)",
            }}
          />
        </button>
      </div>
    </div>
  );
}

// ── Summary row by type ───────────────────────────────────────────

function TypeSummary({ alerts }: { alerts: CognitiveAlert[] }) {
  const counts: Record<string, number> = {};
  for (const a of alerts) {
    if (!a.acknowledged) counts[a.type] = (counts[a.type] || 0) + 1;
  }
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 6);
  if (!sorted.length) return null;

  return (
    <div className="flex flex-wrap gap-2">
      {sorted.map(([type, count]) => (
        <div
          key={type}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#1A2035] border border-[#2A3050] text-[10px]"
        >
          <span className="text-[#64748B]">{TYPE_ICON[type] ?? "!"}</span>
          <span className="text-[#CBD5E1]">{TYPE_LABEL[type] ?? type.replace(/_/g, " ")}</span>
          <span className="text-[#A78BFA] font-bold">{count}</span>
        </div>
      ))}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────

export default function LiveAlertsPage() {
  const [alerts,      setAlerts]      = useState<CognitiveAlert[]>([]);
  const [connected,   setConnected]   = useState(false);
  const [reconnecting,setReconnecting]= useState(false);
  const [filter,      setFilter]      = useState<SeverityFilter>("all");
  const [showAck,     setShowAck]     = useState(false);

  const wsRef         = useRef<WebSocket | null>(null);
  const pingRef       = useRef<ReturnType<typeof setInterval>  | null>(null);
  const reconnectRef  = useRef<ReturnType<typeof setTimeout>   | null>(null);
  const mountedRef    = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    try {
      const token = localStorage.getItem("ptis_access_token");
      const ws = new WebSocket(token ? `${WS_URL}?token=${token}` : WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        setConnected(true);
        setReconnecting(false);
        pingRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "ping" }));
          }
        }, 25_000);
      };

      ws.onmessage = (evt) => {
        if (!mountedRef.current) return;
        try {
          const msg = JSON.parse(evt.data as string);
          // The session feed broadcasts all events — filter cognitive alerts only
          const isCognitiveAlert =
            msg.type === "COGNITIVE_ALERT" ||
            msg.type === "cognitive_alert";

          if (!isCognitiveAlert) return;

          const raw = msg.data?.cognitive_alert ?? msg.data ?? {};
          const alert: CognitiveAlert = {
            id:              `${Date.now()}-${Math.random().toString(36).slice(2)}`,
            type:            raw.type            ?? "unknown",
            severity:        raw.severity        ?? "info",
            title:           raw.title           ?? raw.type ?? "Alert",
            message:         raw.message         ?? "",
            action_required: raw.action_required,
            trade_id:        raw.trade_id,
            symbol:          raw.symbol,
            metadata:        raw.metadata,
            timestamp:       msg.timestamp ?? new Date().toISOString(),
            acknowledged:    false,
          };

          setAlerts((prev) => [alert, ...prev].slice(0, 200));
        } catch {
          // ignore malformed frames
        }
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        setConnected(false);
        if (pingRef.current) clearInterval(pingRef.current);
        setReconnecting(true);
        reconnectRef.current = setTimeout(() => {
          if (mountedRef.current) connect();
        }, 3_000);
      };

      ws.onerror = () => ws.close();
    } catch {
      setConnected(false);
      setReconnecting(true);
      reconnectRef.current = setTimeout(() => {
        if (mountedRef.current) connect();
      }, 5_000);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      if (pingRef.current)      clearInterval(pingRef.current);
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const acknowledge    = (id: string) =>
    setAlerts((prev) => prev.map((a) => (a.id === id ? { ...a, acknowledged: true } : a)));

  const acknowledgeAll = () =>
    setAlerts((prev) => prev.map((a) => ({ ...a, acknowledged: true })));

  const clearAll = () => setAlerts([]);

  // Derived views
  const visible = alerts.filter((a) => {
    if (!showAck && a.acknowledged) return false;
    if (filter !== "all" && a.severity !== filter) return false;
    return true;
  });

  const activeAlerts = alerts.filter((a) => !a.acknowledged);
  const criticalCount = activeAlerts.filter((a) => a.severity === "critical").length;
  const warningCount  = activeAlerts.filter((a) => a.severity === "warning").length;

  return (
    <div className="flex flex-col h-[calc(100vh-56px)]">
      {/* ── Header ──────────────────────────────────────────────── */}
      <div className="px-4 py-3 border-b border-[#2A3050] flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-lg bg-[#EF4444]/20 border border-[#EF4444]/30 flex items-center justify-center">
            <span className="text-[#EF4444] text-sm font-bold">◉</span>
          </div>
          <div>
            <h1 className="text-sm font-bold text-[#F1F5F9]">Live Cognitive Alerts</h1>
            <p className="text-[10px] text-[#64748B]">Real-time discipline · risk · emotion monitoring</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Connection status badge */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-medium ${
              connected
                ? "bg-[#10B981]/10 text-[#10B981]"
                : reconnecting
                  ? "bg-[#F59E0B]/10 text-[#F59E0B]"
                  : "bg-[#EF4444]/10 text-[#EF4444]"
            }`}
          >
            <div
              className={`h-1.5 w-1.5 rounded-full ${
                connected ? "bg-[#10B981] pulse-dot" :
                reconnecting ? "bg-[#F59E0B]" : "bg-[#EF4444]"
              }`}
            />
            {connected ? "Live" : reconnecting ? "Reconnecting…" : "Disconnected"}
          </div>

          {alerts.length > 0 && (
            <button
              onClick={clearAll}
              className="text-[11px] text-[#64748B] hover:text-[#94A3B8] transition-colors"
            >
              Clear all
            </button>
          )}
        </div>
      </div>

      {/* ── Stats bar ────────────────────────────────────────────── */}
      <div className="px-4 py-3 border-b border-[#2A3050] grid grid-cols-4 gap-2 shrink-0">
        {[
          { label: "Active",   count: activeAlerts.length, color: "#CBD5E1" },
          { label: "Critical", count: criticalCount,        color: "#EF4444" },
          { label: "Warning",  count: warningCount,         color: "#F59E0B" },
          { label: "Total",    count: alerts.length,        color: "#64748B" },
        ].map((stat) => (
          <div key={stat.label} className="text-center">
            <div className="text-xl font-bold" style={{ color: stat.color }}>
              {stat.count}
            </div>
            <div className="text-[9px] text-[#64748B] uppercase tracking-wide mt-0.5">
              {stat.label}
            </div>
          </div>
        ))}
      </div>

      {/* ── Type summary ─────────────────────────────────────────── */}
      {activeAlerts.length > 0 && (
        <div className="px-4 py-2 border-b border-[#2A3050] shrink-0">
          <TypeSummary alerts={alerts} />
        </div>
      )}

      {/* ── Filter + actions bar ─────────────────────────────────── */}
      <div className="px-4 py-2 border-b border-[#2A3050] shrink-0 flex items-center gap-2 flex-wrap">
        <FilterBar
          filter={filter}
          setFilter={setFilter}
          showAck={showAck}
          setShowAck={setShowAck}
        />
        {activeAlerts.length > 1 && (
          <button
            onClick={acknowledgeAll}
            className="ml-2 text-[11px] px-3 py-1 rounded-lg border border-[#2A3050] text-[#64748B] hover:text-[#94A3B8] hover:border-[#3A4060] transition-all"
          >
            Acknowledge all
          </button>
        )}
      </div>

      {/* ── Alert feed ───────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {visible.length === 0 && (
          <div className="py-20 text-center">
            {connected ? (
              <>
                <div className="text-3xl mb-3 text-[#10B981]">✓</div>
                <div className="text-sm text-[#64748B]">
                  {activeAlerts.length > 0
                    ? "No alerts match current filter"
                    : "No active alerts — discipline is intact"}
                </div>
                <div className="text-[10px] text-[#3A4060] mt-1">
                  Alerts appear here instantly when triggered
                </div>
              </>
            ) : (
              <>
                <div className="text-3xl mb-3 text-[#F59E0B] animate-pulse">⟳</div>
                <div className="text-sm text-[#64748B]">
                  {reconnecting ? "Reconnecting to alert stream…" : "Connecting…"}
                </div>
                <div className="text-[10px] text-[#3A4060] mt-1">
                  Ensure the backend is running on {WS_URL.replace("ws", "http")}
                </div>
              </>
            )}
          </div>
        )}

        {visible.map((alert) => (
          <AlertCard key={alert.id} alert={alert} onAcknowledge={acknowledge} />
        ))}
      </div>
    </div>
  );
}
