// src/pages/SettingsPage.tsx
import { useState, useEffect, useCallback } from "react";
import { tokenStore, brokerApi, BrokerStatusResponse, BrokerInfo } from "@/services/api";

const CARD = "bg-[#1A2035] border border-[#2A3050] rounded-lg p-5";
const INPUT = "w-full bg-[#0F1629] border border-[#2A3050] text-[#F1F5F9] text-sm rounded px-3 py-2 focus:outline-none focus:border-[#8B5CF6]";
const LABEL = "block text-[10px] text-[#64748B] uppercase tracking-wide mb-1";
const TOGGLE_ON = "relative w-9 h-5 bg-[#8B5CF6] rounded-full cursor-pointer transition-colors";
const TOGGLE_OFF = "relative w-9 h-5 bg-[#2A3050] rounded-full cursor-pointer transition-colors";

interface UserSettings {
  // Risk
  maxRiskPerTrade: string;
  maxOpenPositions: string;
  dailyLossLimit: string;
  maxDailyTrades: string;
  // Notifications
  revengeTradeAlert: boolean;
  lossLimitWarning: boolean;
  sessionTimeLimit: boolean;
  morningBriefing: boolean;
  // Coaching
  coachingPersonality: string;
  feedbackStyle: string;
  // Display
  theme: string;
}

const DEFAULT_SETTINGS: UserSettings = {
  maxRiskPerTrade: "1",
  maxOpenPositions: "5",
  dailyLossLimit: "3",
  maxDailyTrades: "8",
  revengeTradeAlert: true,
  lossLimitWarning: true,
  sessionTimeLimit: false,
  morningBriefing: true,
  coachingPersonality: "balanced",
  feedbackStyle: "direct",
  theme: "dark",
};

function loadSettings(): UserSettings {
  try {
    const raw = localStorage.getItem("ptis_settings");
    if (raw) return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };

    // Fallback: merge from onboarding data
    const onb = localStorage.getItem("ptis_onboarding");
    if (onb) {
      const d = JSON.parse(onb);
      return {
        ...DEFAULT_SETTINGS,
        maxRiskPerTrade: d.maxRiskPerTrade || DEFAULT_SETTINGS.maxRiskPerTrade,
        maxOpenPositions: d.maxOpenPositions || DEFAULT_SETTINGS.maxOpenPositions,
      };
    }
  } catch {}
  return { ...DEFAULT_SETTINGS };
}

function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button onClick={() => onChange(!value)} className={value ? TOGGLE_ON : TOGGLE_OFF}>
      <div
        className="absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform"
        style={{ left: value ? "18px" : "2px" }}
      />
    </button>
  );
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<UserSettings>(loadSettings);
  const [saved, setSaved] = useState(false);
  const user = tokenStore.getUser();

  // Broker status
  const [brokerStatus, setBrokerStatus] = useState<BrokerStatusResponse | null>(null);
  const [brokerLoading, setBrokerLoading] = useState(true);
  const [testResult, setTestResult] = useState<Record<string, { status: string; detail?: string }>>({});
  const [syncing, setSyncing] = useState(false);

  const fetchBrokerStatus = useCallback(async () => {
    try {
      const res = await brokerApi.status();
      setBrokerStatus(res);
    } catch {
      setBrokerStatus(null);
    } finally {
      setBrokerLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchBrokerStatus();
  }, [fetchBrokerStatus]);

  const handleTestAngel = async () => {
    setTestResult((p) => ({ ...p, angel: { status: "testing..." } }));
    try {
      const res = await brokerApi.testAngel();
      setTestResult((p) => ({
        ...p,
        angel: {
          status: res.status,
          detail: res.candles_fetched ? `${res.candles_fetched} candles, last close ₹${res.last_close}` : res.detail,
        },
      }));
    } catch (e: any) {
      setTestResult((p) => ({ ...p, angel: { status: "error", detail: e.message || "Failed" } }));
    }
  };

  const handleTestUpstox = async () => {
    setTestResult((p) => ({ ...p, upstox: { status: "testing..." } }));
    try {
      const res = await brokerApi.testUpstox();
      setTestResult((p) => ({
        ...p,
        upstox: { status: res.status, detail: res.ltp ? `RELIANCE ₹${res.ltp}` : res.detail },
      }));
    } catch (e: any) {
      setTestResult((p) => ({ ...p, upstox: { status: "error", detail: e.message || "Failed" } }));
    }
  };

  const handleConnectUpstox = async () => {
    try {
      const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const res = await fetch(`${BASE_URL}/api/v1/broker/upstox/login-url`, {
        headers: { Authorization: `Bearer ${tokenStore.getToken() ?? "dev"}` },
      });
      const data = await res.json();
      if (data.login_url) {
        window.open(data.login_url, "_blank", "width=500,height=700");
        // Poll broker status after 10s to reflect the new connection
        setTimeout(fetchBrokerStatus, 10000);
      }
    } catch (e: any) {
      setTestResult((p) => ({ ...p, upstox: { status: "error", detail: "Could not get login URL" } }));
    }
  };

  const handleTestGroww = async () => {
    setTestResult((p) => ({ ...p, groww: { status: "testing..." } }));
    try {
      const res = await brokerApi.testGroww();
      setTestResult((p) => ({
        ...p,
        groww: { status: res.status, detail: res.account ? `Equity ₹${res.account.equity}` : res.detail },
      }));
    } catch (e: any) {
      setTestResult((p) => ({ ...p, groww: { status: "error", detail: e.message || "Failed" } }));
    }
  };

  const handleSyncNow = async () => {
    setSyncing(true);
    try {
      const res = await brokerApi.triggerSync();
      setTestResult((p) => ({
        ...p,
        sync: { status: "done", detail: `${res.synced} synced, ${res.skipped} skipped` },
      }));
      fetchBrokerStatus();
    } catch (e: any) {
      setTestResult((p) => ({ ...p, sync: { status: "error", detail: e.message || "Failed" } }));
    } finally {
      setSyncing(false);
    }
  };

  const renderBrokerRow = (label: string, info: BrokerInfo, key: string, onTest: () => void) => (
    <div className="flex items-center justify-between py-2 border-b border-[#2A3050]/50">
      <div className="flex items-center gap-2">
        <div className={`h-2 w-2 rounded-full ${info.connected ? "bg-[#22C55E]" : info.configured ? "bg-[#F59E0B]" : "bg-[#64748B]"}`} />
        <div>
          <span className="text-xs text-[#CBD5E1]">{label}</span>
          <span className="text-[10px] text-[#64748B] ml-2">{info.role}</span>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <span className={`text-[10px] px-2 py-0.5 rounded border ${
          info.connected
            ? "bg-[#22C55E]/10 border-[#22C55E]/30 text-[#22C55E]"
            : "bg-[#0F1629] border-[#2A3050] text-[#64748B]"
        }`}>
          {info.connected ? "Connected" : info.configured ? "Configured" : "Not Configured"}
        </span>
        {info.configured && (
          <button
            onClick={onTest}
            className="text-[10px] px-2 py-0.5 rounded border border-[#2A3050] text-[#8B5CF6] hover:bg-[#8B5CF6]/10 transition-colors"
          >
            Test
          </button>
        )}
      </div>
    </div>
  );

  const renderTestBadge = (result: { status: string; detail?: string }) => (
    <div className={`text-[10px] px-3 py-1.5 rounded ${
      result.status === "connected" || result.status === "done" ? "bg-[#22C55E]/10 text-[#22C55E]" :
      result.status === "testing..." ? "bg-[#8B5CF6]/10 text-[#8B5CF6]" :
      "bg-[#EF4444]/10 text-[#F87171]"
    }`}>
      {result.detail || result.status}
    </div>
  );

  const update = <K extends keyof UserSettings>(key: K, value: UserSettings[K]) => {
    setSettings((s) => ({ ...s, [key]: value }));
    setSaved(false);
  };

  const save = () => {
    localStorage.setItem("ptis_settings", JSON.stringify(settings));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="p-4 max-w-[700px] mx-auto animate-fade-in">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h1 className="text-xl font-bold text-[#F1F5F9]">Settings</h1>
          <p className="text-xs text-[#64748B] mt-0.5">Configure your PTIS experience</p>
        </div>
        <button
          onClick={save}
          className="px-4 py-2 bg-[#8B5CF6] hover:bg-[#7C3AED] text-white text-xs font-medium rounded-lg transition-colors"
        >
          {saved ? "Saved" : "Save Changes"}
        </button>
      </div>

      <div className="space-y-4">
        {/* Account Info */}
        <div className={CARD}>
          <h3 className="text-sm font-semibold text-[#F1F5F9] mb-3">Account</h3>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className={LABEL}>Username</label>
              <div className="text-sm text-[#94A3B8]">{user?.username || "—"}</div>
            </div>
            <div>
              <label className={LABEL}>Email</label>
              <div className="text-sm text-[#94A3B8]">{user?.email || "—"}</div>
            </div>
          </div>
        </div>

        {/* Risk Rules */}
        <div className={CARD}>
          <h3 className="text-sm font-semibold text-[#F1F5F9] mb-0.5">Risk Rules</h3>
          <p className="text-xs text-[#64748B] mb-3">Hard limits to protect your capital</p>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className={LABEL}>Max Risk Per Trade (%)</label>
              <input
                type="number" step="0.1" min="0.1" max="5"
                value={settings.maxRiskPerTrade}
                onChange={(e) => update("maxRiskPerTrade", e.target.value)}
                className={INPUT}
              />
            </div>
            <div>
              <label className={LABEL}>Max Open Positions</label>
              <input
                type="number" min="1" max="50"
                value={settings.maxOpenPositions}
                onChange={(e) => update("maxOpenPositions", e.target.value)}
                className={INPUT}
              />
            </div>
            <div>
              <label className={LABEL}>Daily Loss Limit (%)</label>
              <input
                type="number" step="0.5" min="0.5" max="10"
                value={settings.dailyLossLimit}
                onChange={(e) => update("dailyLossLimit", e.target.value)}
                className={INPUT}
              />
            </div>
            <div>
              <label className={LABEL}>Max Trades Per Day</label>
              <input
                type="number" min="1" max="50"
                value={settings.maxDailyTrades}
                onChange={(e) => update("maxDailyTrades", e.target.value)}
                className={INPUT}
              />
            </div>
          </div>
        </div>

        {/* Notifications */}
        <div className={CARD}>
          <h3 className="text-sm font-semibold text-[#F1F5F9] mb-0.5">Notifications</h3>
          <p className="text-xs text-[#64748B] mb-3">Behavioral alerts and coaching nudges</p>
          <div className="space-y-3">
            {([
              { key: "revengeTradeAlert" as const, label: "Revenge Trade Alert", desc: "Alert when entering a trade too quickly after a loss" },
              { key: "lossLimitWarning" as const, label: "Loss Limit Warning", desc: "Warn when approaching daily loss limit" },
              { key: "sessionTimeLimit" as const, label: "Session Time Limit", desc: "Remind to take breaks after extended sessions" },
              { key: "morningBriefing" as const, label: "Morning Briefing", desc: "Daily pre-session market context summary" },
            ]).map(({ key, label, desc }) => (
              <div key={key} className="flex items-center justify-between py-1.5 border-b border-[#2A3050]/50 last:border-0">
                <div>
                  <div className="text-xs font-medium text-[#CBD5E1]">{label}</div>
                  <div className="text-[10px] text-[#64748B]">{desc}</div>
                </div>
                <Toggle value={settings[key]} onChange={(v) => update(key, v)} />
              </div>
            ))}
          </div>
        </div>

        {/* AI Coaching */}
        <div className={CARD}>
          <h3 className="text-sm font-semibold text-[#F1F5F9] mb-0.5">AI Coaching</h3>
          <p className="text-xs text-[#64748B] mb-3">Personalize your coaching experience</p>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className={LABEL}>Coaching Personality</label>
              <select
                value={settings.coachingPersonality}
                onChange={(e) => update("coachingPersonality", e.target.value)}
                className={INPUT}
              >
                <option value="balanced">Balanced</option>
                <option value="strict">Strict Disciplinarian</option>
                <option value="encouraging">Encouraging Mentor</option>
                <option value="analytical">Pure Analytical</option>
              </select>
            </div>
            <div>
              <label className={LABEL}>Feedback Style</label>
              <select
                value={settings.feedbackStyle}
                onChange={(e) => update("feedbackStyle", e.target.value)}
                className={INPUT}
              >
                <option value="direct">Direct & Concise</option>
                <option value="detailed">Detailed Explanations</option>
                <option value="socratic">Socratic (Questions)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Broker Integration */}
        <div className={CARD}>
          <h3 className="text-sm font-semibold text-[#F1F5F9] mb-0.5">Broker Integration</h3>
          <p className="text-xs text-[#64748B] mb-3">Historical data, live market feed, and trade import</p>

          {brokerLoading ? (
            <div className="text-xs text-[#64748B] py-4 text-center">Loading broker status...</div>
          ) : !brokerStatus ? (
            <div className="text-xs text-[#64748B] py-4 text-center">Could not fetch broker status</div>
          ) : (
            <div className="space-y-3">
              {/* Angel One — Historical Data */}
              {renderBrokerRow("Angel One (SmartAPI)", brokerStatus.angel_one, "angel", handleTestAngel)}
              {testResult.angel && renderTestBadge(testResult.angel)}

              {/* Upstox — Live Market Data */}
              {renderBrokerRow("Upstox", brokerStatus.upstox, "upstox", handleTestUpstox)}
              {testResult.upstox && renderTestBadge(testResult.upstox)}
              <div className="flex items-center gap-2 pl-4">
                <button
                  onClick={handleConnectUpstox}
                  className="text-[10px] px-3 py-1 rounded bg-[#3B82F6]/10 border border-[#3B82F6]/30 text-[#3B82F6] hover:bg-[#3B82F6]/20 transition-colors"
                >
                  {brokerStatus.upstox.connected ? "Re-connect Upstox" : "Connect Upstox"}
                </button>
                <span className="text-[10px] text-[#64748B]">
                  {brokerStatus.upstox.connected ? "Token active — refresh daily" : "Opens Upstox login in new tab"}
                </span>
              </div>

              {/* Groww — Trade Tracking */}
              {renderBrokerRow("Groww", brokerStatus.groww, "groww", handleTestGroww)}
              {testResult.groww && renderTestBadge(testResult.groww)}

              {/* Sync Controls */}
              {brokerStatus.groww.connected && (
                <div className="flex items-center justify-between pt-2">
                  <div className="text-[10px] text-[#64748B]">
                    {brokerStatus.groww.last_sync
                      ? `Last sync: ${new Date(brokerStatus.groww.last_sync).toLocaleString()} (${brokerStatus.groww.sync_count} cycles)`
                      : "No sync yet"}
                  </div>
                  <button
                    onClick={handleSyncNow}
                    disabled={syncing}
                    className="text-[10px] px-3 py-1 rounded bg-[#8B5CF6]/10 border border-[#8B5CF6]/30 text-[#8B5CF6] hover:bg-[#8B5CF6]/20 transition-colors disabled:opacity-50"
                  >
                    {syncing ? "Syncing..." : "Sync Now"}
                  </button>
                </div>
              )}
              {testResult.sync && renderTestBadge(testResult.sync)}

              <p className="text-[10px] text-[#475569] mt-1">
                Broker connections are configured via .env variables on the server.
              </p>
            </div>
          )}
        </div>

        {/* Danger Zone */}
        <div className="bg-[#1A2035] border border-[#EF4444]/20 rounded-lg p-5">
          <h3 className="text-sm font-semibold text-[#F87171] mb-0.5">Danger Zone</h3>
          <p className="text-xs text-[#64748B] mb-3">Irreversible actions</p>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-[#CBD5E1]">Reset All Settings</div>
              <div className="text-[10px] text-[#64748B]">Restore all settings to their defaults</div>
            </div>
            <button
              onClick={() => {
                if (confirm("Reset all settings to defaults?")) {
                  localStorage.removeItem("ptis_settings");
                  setSettings({ ...DEFAULT_SETTINGS });
                  setSaved(false);
                }
              }}
              className="text-[10px] px-3 py-1.5 rounded border border-[#EF4444]/30 text-[#F87171] hover:bg-[#EF4444]/10 transition-colors"
            >
              Reset
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
