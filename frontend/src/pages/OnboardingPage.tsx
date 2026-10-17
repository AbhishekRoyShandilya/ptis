// src/pages/OnboardingPage.tsx
// 5-step onboarding wizard for new PTIS users
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { authApi, tokenStore } from "@/services/api";

const CARD = "bg-[#1A2035] border border-[#2A3050] rounded-lg p-6";

type Step = 1 | 2 | 3 | 4 | 5;

const STEPS = [
  { num: 1, label: "Welcome" },
  { num: 2, label: "Experience" },
  { num: 3, label: "Risk Rules" },
  { num: 4, label: "Strategy" },
  { num: 5, label: "Watchlist" },
];

interface OnboardingData {
  // Step 1: Welcome
  displayName: string;
  // Step 2: Experience
  experience: string;
  tradingStyle: string;
  marketsTraded: string[];
  // Step 3: Risk
  accountSize: string;
  maxRiskPerTrade: string;
  maxOpenPositions: string;
  // Step 4: Strategy
  strategyChoice: "teach" | "discover" | "bootstrap" | "skip";
  // Step 5: Watchlist
  watchlistStocks: string;
}

export default function OnboardingPage() {
  const navigate = useNavigate();
  const [step, setStep] = useState<Step>(1);
  const [data, setData] = useState<OnboardingData>({
    displayName: tokenStore.getUser()?.full_name || "",
    experience: "",
    tradingStyle: "",
    marketsTraded: [],
    accountSize: "500000",
    maxRiskPerTrade: "1",
    maxOpenPositions: "5",
    strategyChoice: "skip",
    watchlistStocks: "",
  });

  const update = <K extends keyof OnboardingData>(key: K, value: OnboardingData[K]) =>
    setData((d) => ({ ...d, [key]: value }));

  const next = () => {
    if (step < 5) setStep((s) => (s + 1) as Step);
  };
  const prev = () => {
    if (step > 1) setStep((s) => (s - 1) as Step);
  };

  const finish = async () => {
    // Persist onboarding data to backend
    try {
      const updated = await authApi.updateProfile({
        onboarding_data: data as unknown as Record<string, unknown>,
        onboarding_completed: true,
      });
      tokenStore.setUser(updated);
    } catch {
      // Fallback to localStorage if backend fails
      localStorage.setItem("ptis_onboarding", JSON.stringify(data));
      const user = tokenStore.getUser();
      if (user) {
        tokenStore.setUser({ ...user, onboarding_completed: true });
      }
    }

    // Navigate based on strategy choice. "teach" and "bootstrap" are now the
    // same unified Teach tab (method picker chooses charts vs describe).
    if (data.strategyChoice === "discover") {
      navigate("/strategies?tab=discovery");
    } else if (data.strategyChoice === "teach" || data.strategyChoice === "bootstrap") {
      navigate("/strategies?tab=teach");
    } else {
      navigate("/");
    }
  };

  const progress = (step / 5) * 100;

  return (
    <div className="min-h-screen bg-[#0B0F1A] flex items-center justify-center px-4 py-8">
      <div className="w-full max-w-lg">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-xl font-bold text-[#F1F5F9]">Welcome to PTIS</h1>
          <p className="text-xs text-[#64748B] mt-1">Let's set up your trading intelligence system</p>
        </div>

        {/* Progress */}
        <div className="mb-6">
          <div className="flex justify-between mb-2">
            {STEPS.map((s) => (
              <div key={s.num} className="flex flex-col items-center">
                <div className={`w-7 h-7 rounded-full flex items-center justify-center text-[10px] font-bold border-2 transition-colors ${
                  step >= s.num
                    ? "bg-[#8B5CF6] border-[#8B5CF6] text-white"
                    : "border-[#2A3050] text-[#64748B]"
                }`}>{s.num}</div>
                <span className={`text-[9px] mt-1 ${step >= s.num ? "text-[#C4B5FD]" : "text-[#64748B]"}`}>{s.label}</span>
              </div>
            ))}
          </div>
          <div className="h-1 bg-[#1A2035] rounded-full overflow-hidden">
            <div className="h-full bg-[#8B5CF6] rounded-full transition-all" style={{ width: `${progress}%` }} />
          </div>
        </div>

        {/* Step Content */}
        <div className={CARD}>
          {step === 1 && <Step1Welcome data={data} update={update} />}
          {step === 2 && <Step2Experience data={data} update={update} />}
          {step === 3 && <Step3Risk data={data} update={update} />}
          {step === 4 && <Step4Strategy data={data} update={update} />}
          {step === 5 && <Step5Watchlist data={data} update={update} />}

          {/* Navigation */}
          <div className="flex justify-between mt-6 pt-4 border-t border-[#2A3050]">
            {step > 1 ? (
              <button onClick={prev}
                className="text-xs text-[#64748B] hover:text-[#94A3B8] px-4 py-2">Back</button>
            ) : <div />}

            {step < 5 ? (
              <button onClick={next}
                className="text-xs bg-[#8B5CF6] hover:bg-[#7C3AED] text-white px-6 py-2 rounded font-medium">
                Continue
              </button>
            ) : (
              <button onClick={finish}
                className="text-xs bg-[#10B981] hover:bg-[#059669] text-white px-6 py-2 rounded font-medium">
                Get Started
              </button>
            )}
          </div>
        </div>

        {/* Skip */}
        <div className="text-center mt-4">
          <button onClick={finish} className="text-[10px] text-[#64748B] hover:text-[#94A3B8]">
            Skip setup and go to dashboard
          </button>
        </div>
      </div>
    </div>
  );
}


// ── Step Components ──────────────────────────────────────────────

function Step1Welcome({ data, update }: StepProps) {
  return (
    <div>
      <h3 className="text-sm font-bold text-[#F1F5F9] mb-1">About You</h3>
      <p className="text-[10px] text-[#64748B] mb-4">
        PTIS adapts to your trading style. This helps us personalize your experience.
      </p>

      <div>
        <label className="block text-[10px] text-[#64748B] uppercase mb-1">What should we call you?</label>
        <input type="text" value={data.displayName}
          onChange={(e) => update("displayName", e.target.value)}
          className="w-full bg-[#0F1629] border border-[#2A3050] text-[#F1F5F9] text-sm rounded px-3 py-2 focus:outline-none focus:border-[#8B5CF6]"
          placeholder="Your name" />
      </div>

      <div className="mt-6 bg-[#0F1629] rounded-lg p-4">
        <h4 className="text-xs font-medium text-[#C4B5FD] mb-2">What PTIS will do for you</h4>
        <ul className="space-y-2 text-[11px] text-[#94A3B8]">
          <li className="flex gap-2"><span className="text-[#10B981]">1.</span> Track your behavior — know if you're following your strategies</li>
          <li className="flex gap-2"><span className="text-[#3B82F6]">2.</span> Co-plan trades — AI analysis before you enter</li>
          <li className="flex gap-2"><span className="text-[#A78BFA]">3.</span> AVT trades your strategies virtually — finds improvements</li>
          <li className="flex gap-2"><span className="text-[#F59E0B]">4.</span> Discover patterns and make strategies better</li>
          <li className="flex gap-2"><span className="text-[#EF4444]">5.</span> Real-time alerts and AI coaching</li>
        </ul>
      </div>
    </div>
  );
}


function Step2Experience({ data, update }: StepProps) {
  const experiences = [
    { key: "beginner", label: "Beginner", desc: "< 1 year of active trading" },
    { key: "intermediate", label: "Intermediate", desc: "1-3 years, have defined strategies" },
    { key: "advanced", label: "Advanced", desc: "3+ years, consistent edge" },
    { key: "professional", label: "Professional", desc: "Full-time trader or prop desk" },
  ];

  const styles = [
    { key: "scalper", label: "Scalper", desc: "Minutes, quick in and out" },
    { key: "intraday", label: "Intraday", desc: "Same-day, close by market end" },
    { key: "swing", label: "Swing", desc: "Days to weeks" },
    { key: "positional", label: "Positional", desc: "Weeks to months" },
  ];

  const markets = [
    { key: "equity_cash", label: "Equity Cash" },
    { key: "equity_fno", label: "Equity F&O" },
    { key: "index_options", label: "Index Options" },
    { key: "commodities", label: "Commodities" },
    { key: "forex", label: "Forex" },
  ];

  const toggleMarket = (key: string) => {
    const current = data.marketsTraded;
    update("marketsTraded", current.includes(key) ? current.filter((m) => m !== key) : [...current, key]);
  };

  return (
    <div>
      <h3 className="text-sm font-bold text-[#F1F5F9] mb-1">Trading Experience</h3>
      <p className="text-[10px] text-[#64748B] mb-4">This helps PTIS calibrate its coaching and analysis depth.</p>

      <div className="mb-4">
        <label className="block text-[10px] text-[#64748B] uppercase mb-2">Experience Level</label>
        <div className="grid grid-cols-2 gap-2">
          {experiences.map((exp) => (
            <button key={exp.key} onClick={() => update("experience", exp.key)}
              className={`text-left px-3 py-2 rounded border text-xs transition-colors ${
                data.experience === exp.key
                  ? "border-[#8B5CF6] bg-[#8B5CF6]/10 text-[#C4B5FD]"
                  : "border-[#2A3050] text-[#94A3B8] hover:border-[#3A4060]"
              }`}>
              <div className="font-medium">{exp.label}</div>
              <div className="text-[9px] opacity-70 mt-0.5">{exp.desc}</div>
            </button>
          ))}
        </div>
      </div>

      <div className="mb-4">
        <label className="block text-[10px] text-[#64748B] uppercase mb-2">Primary Style</label>
        <div className="grid grid-cols-2 gap-2">
          {styles.map((s) => (
            <button key={s.key} onClick={() => update("tradingStyle", s.key)}
              className={`text-left px-3 py-2 rounded border text-xs transition-colors ${
                data.tradingStyle === s.key
                  ? "border-[#8B5CF6] bg-[#8B5CF6]/10 text-[#C4B5FD]"
                  : "border-[#2A3050] text-[#94A3B8] hover:border-[#3A4060]"
              }`}>
              <div className="font-medium">{s.label}</div>
              <div className="text-[9px] opacity-70 mt-0.5">{s.desc}</div>
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="block text-[10px] text-[#64748B] uppercase mb-2">Markets You Trade</label>
        <div className="flex flex-wrap gap-2">
          {markets.map((m) => (
            <button key={m.key} onClick={() => toggleMarket(m.key)}
              className={`px-3 py-1.5 rounded-full text-[10px] font-medium border transition-colors ${
                data.marketsTraded.includes(m.key)
                  ? "border-[#8B5CF6] bg-[#8B5CF6]/10 text-[#C4B5FD]"
                  : "border-[#2A3050] text-[#64748B] hover:border-[#3A4060]"
              }`}>{m.label}</button>
          ))}
        </div>
      </div>
    </div>
  );
}


function Step3Risk({ data, update }: StepProps) {
  return (
    <div>
      <h3 className="text-sm font-bold text-[#F1F5F9] mb-1">Risk Preferences</h3>
      <p className="text-[10px] text-[#64748B] mb-4">
        PTIS uses these to calibrate position sizing, alerts, and AVT parameters. You can change them later.
      </p>

      <div className="space-y-4">
        <div>
          <label className="block text-[10px] text-[#64748B] uppercase mb-1">Account Size (INR)</label>
          <div className="flex gap-2">
            {["200000", "500000", "1000000", "2500000", "5000000"].map((v) => (
              <button key={v} onClick={() => update("accountSize", v)}
                className={`flex-1 px-2 py-2 rounded border text-xs transition-colors ${
                  data.accountSize === v
                    ? "border-[#8B5CF6] bg-[#8B5CF6]/10 text-[#C4B5FD]"
                    : "border-[#2A3050] text-[#64748B] hover:border-[#3A4060]"
                }`}>
                {(Number(v) / 100000).toFixed(0)}L
              </button>
            ))}
          </div>
          <input type="number" value={data.accountSize}
            onChange={(e) => update("accountSize", e.target.value)}
            className="w-full mt-2 bg-[#0F1629] border border-[#2A3050] text-[#F1F5F9] text-sm rounded px-3 py-2 focus:outline-none focus:border-[#8B5CF6]"
            placeholder="Custom amount" />
        </div>

        <div>
          <label className="block text-[10px] text-[#64748B] uppercase mb-1">Max Risk Per Trade (%)</label>
          <div className="flex gap-2">
            {["0.5", "1", "1.5", "2"].map((v) => (
              <button key={v} onClick={() => update("maxRiskPerTrade", v)}
                className={`flex-1 px-3 py-2 rounded border text-xs transition-colors ${
                  data.maxRiskPerTrade === v
                    ? "border-[#8B5CF6] bg-[#8B5CF6]/10 text-[#C4B5FD]"
                    : "border-[#2A3050] text-[#64748B] hover:border-[#3A4060]"
                }`}>{v}%</button>
            ))}
          </div>
          <p className="text-[9px] text-[#64748B] mt-1">Recommended: 1% for most traders, 0.5% if conservative</p>
        </div>

        <div>
          <label className="block text-[10px] text-[#64748B] uppercase mb-1">Max Simultaneous Positions</label>
          <div className="flex gap-2">
            {["3", "5", "8", "10"].map((v) => (
              <button key={v} onClick={() => update("maxOpenPositions", v)}
                className={`flex-1 px-3 py-2 rounded border text-xs transition-colors ${
                  data.maxOpenPositions === v
                    ? "border-[#8B5CF6] bg-[#8B5CF6]/10 text-[#C4B5FD]"
                    : "border-[#2A3050] text-[#64748B] hover:border-[#3A4060]"
                }`}>{v}</button>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-4 bg-[#0F1629] rounded-lg p-3">
        <div className="text-[10px] text-[#64748B]">With these settings:</div>
        <div className="text-xs text-[#F1F5F9] mt-1">
          Max risk per trade: <span className="text-[#A78BFA] font-bold">
            {(Number(data.accountSize) * Number(data.maxRiskPerTrade) / 100).toLocaleString("en-IN")} INR
          </span>
        </div>
        <div className="text-xs text-[#F1F5F9]">
          Total portfolio heat: <span className="text-[#F59E0B] font-bold">
            {(Number(data.maxRiskPerTrade) * Number(data.maxOpenPositions)).toFixed(1)}%
          </span>
          <span className="text-[9px] text-[#64748B] ml-1">
            ({Number(data.maxRiskPerTrade)}% x {data.maxOpenPositions} positions)
          </span>
        </div>
      </div>
    </div>
  );
}


function Step4Strategy({ data, update }: StepProps) {
  const options = [
    {
      key: "teach" as const,
      icon: "T",
      color: "#8B5CF6",
      title: "Teach Your Strategy",
      desc: "Upload chart screenshots and annotate your entries/exits. PTIS learns your edge.",
      detail: "Best for: You have a proven strategy you trade consistently",
    },
    {
      key: "discover" as const,
      icon: "D",
      color: "#3B82F6",
      title: "Discover Patterns",
      desc: "Describe a price pattern in plain English. PTIS finds it across 200+ NSE stocks.",
      detail: "Best for: You see patterns but haven't formalized them yet",
    },
    {
      key: "bootstrap" as const,
      icon: "B",
      color: "#10B981",
      title: "Describe Your Strategy",
      desc: "Tell PTIS about your strategy in plain English. It interprets, backtests, and learns it instantly.",
      detail: "Best for: You know your strategy but don't have screenshots ready",
    },
    {
      key: "skip" as const,
      icon: "S",
      color: "#64748B",
      title: "I'll Do This Later",
      desc: "Go straight to the dashboard. You can teach strategies or discover patterns anytime.",
      detail: "You can always come back to Learn or Discovery from the nav bar",
    },
  ];

  return (
    <div>
      <h3 className="text-sm font-bold text-[#F1F5F9] mb-1">Your First Strategy</h3>
      <p className="text-[10px] text-[#64748B] mb-4">
        PTIS needs to understand your trading strategies to help you. How would you like to start?
      </p>

      <div className="space-y-3">
        {options.map((opt) => (
          <button key={opt.key} onClick={() => update("strategyChoice", opt.key)}
            className={`w-full text-left px-4 py-4 rounded-lg border transition-colors ${
              data.strategyChoice === opt.key
                ? "border-[#8B5CF6] bg-[#8B5CF6]/5"
                : "border-[#2A3050] hover:border-[#3A4060]"
            }`}>
            <div className="flex gap-3 items-start">
              <div className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold shrink-0"
                style={{ backgroundColor: opt.color + "20", color: opt.color }}>
                {opt.icon}
              </div>
              <div>
                <div className="text-xs font-bold text-[#F1F5F9]">{opt.title}</div>
                <div className="text-[10px] text-[#94A3B8] mt-0.5">{opt.desc}</div>
                <div className="text-[9px] text-[#64748B] mt-1">{opt.detail}</div>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}


function Step5Watchlist({ data, update }: StepProps) {
  const popularStocks = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
    "AXISBANK", "WIPRO", "TATAMOTORS", "BAJFINANCE", "MARUTI",
    "SUNPHARMA", "TITAN", "ADANIENT", "TATASTEEL", "HINDUNILVR",
  ];

  const toggleStock = (stock: string) => {
    const current = data.watchlistStocks.split(",").map((s) => s.trim()).filter(Boolean);
    const updated = current.includes(stock)
      ? current.filter((s) => s !== stock)
      : [...current, stock];
    update("watchlistStocks", updated.join(", "));
  };

  const selectedStocks = data.watchlistStocks.split(",").map((s) => s.trim()).filter(Boolean);

  return (
    <div>
      <h3 className="text-sm font-bold text-[#F1F5F9] mb-1">Your Watchlist</h3>
      <p className="text-[10px] text-[#64748B] mb-4">
        Select stocks you trade or want to track. AVT will scan these for setups.
      </p>

      <div className="mb-4">
        <label className="block text-[10px] text-[#64748B] uppercase mb-1">
          Custom Stocks (comma separated)
        </label>
        <input type="text" value={data.watchlistStocks}
          onChange={(e) => update("watchlistStocks", e.target.value)}
          className="w-full bg-[#0F1629] border border-[#2A3050] text-[#F1F5F9] text-sm rounded px-3 py-2 focus:outline-none focus:border-[#8B5CF6]"
          placeholder="RELIANCE, TCS, INFY..." />
      </div>

      <div>
        <label className="block text-[10px] text-[#64748B] uppercase mb-2">
          Popular FNO Stocks ({selectedStocks.length} selected)
        </label>
        <div className="flex flex-wrap gap-1.5">
          {popularStocks.map((stock) => (
            <button key={stock} onClick={() => toggleStock(stock)}
              className={`px-2.5 py-1 rounded text-[10px] font-medium border transition-colors ${
                selectedStocks.includes(stock)
                  ? "border-[#8B5CF6] bg-[#8B5CF6]/10 text-[#C4B5FD]"
                  : "border-[#2A3050] text-[#64748B] hover:border-[#3A4060] hover:text-[#94A3B8]"
              }`}>{stock}</button>
          ))}
        </div>
      </div>

      {selectedStocks.length > 0 && (
        <div className="mt-4 bg-[#0F1629] rounded-lg p-3">
          <div className="text-[10px] text-[#64748B]">Your watchlist ({selectedStocks.length} stocks)</div>
          <div className="text-xs text-[#C4B5FD] mt-1">{selectedStocks.join(", ")}</div>
        </div>
      )}

      <div className="mt-4 bg-[#10B981]/10 border border-[#10B981]/30 rounded-lg p-3">
        <div className="text-xs font-medium text-[#34D399]">You're all set!</div>
        <div className="text-[10px] text-[#94A3B8] mt-0.5">
          Click "Get Started" to enter your dashboard. You can always change these settings later.
        </div>
      </div>
    </div>
  );
}


// ── Types ────────────────────────────────────────────────────────

interface StepProps {
  data: OnboardingData;
  update: <K extends keyof OnboardingData>(key: K, value: OnboardingData[K]) => void;
}
