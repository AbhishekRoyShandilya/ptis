// src/store/coPlannerStore.ts
import { create } from "zustand";
import { chatApi, watchlistApi, intelligenceApi, type ChatMessageResponse, type AgentInsight } from "@/services/api";

// Build WebSocket base URL from the Vite API URL
const WS_BASE = (import.meta.env.VITE_API_URL ?? "http://localhost:8000")
  .replace(/^http/, "ws");

// ── Types ───────────────────────────────────────────────────────────

export interface WatchlistStock {
  id: string;
  symbol: string;
  matchScore: number;
  matchReason: string;
  lastPrice?: number;
  changePct?: number;
  status: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  insights?: AgentInsight[];
  actionItems?: string[];
  followUpQuestions?: string[];
  timestamp: Date;
  attachments?: { name: string; type: string; url?: string }[];
}

export interface SessionOpportunity {
  symbol: string;
  grade: "A" | "B" | "C" | "skip";
  setup_type: string | null;
  strategy_id: string | null;
  regime: string;
  regime_fit: string;
  estimated_r: number;
  confidence: number;
  notes: string;
}

export interface SessionBriefingReport {
  generated_at: string;
  top_opportunities: SessionOpportunity[];
  skipped_symbols: { symbol: string; reason: string }[];
  strategies_ready: string[];
  strategies_avoid: string[];
  behavioral_flags: string[];
  session_limit_reached: boolean;
  battle_plan: string;
  plan_generated: boolean;
}

interface CoPlannerState {
  // Watchlist
  activeSymbol: string | null;
  watchlistStocks: WatchlistStock[];
  watchlistLoading: boolean;
  watchlistSearch: string;

  // Chat
  messages: ChatMessage[];
  sessionId: string | null;
  chatLoading: boolean;
  attachments: File[];

  // Proactive alerts (SETUP_DETECTED WebSocket)
  alertWs: WebSocket | null;

  // Session briefing
  briefing: SessionBriefingReport | null;
  briefingLoading: boolean;

  // Actions
  setActiveSymbol: (symbol: string | null) => void;
  setWatchlistSearch: (q: string) => void;
  loadWatchlist: () => Promise<void>;
  sendMessage: (text: string) => Promise<void>;
  addAttachment: (file: File) => void;
  removeAttachment: (index: number) => void;
  clearAttachments: () => void;
  clearChat: () => void;
  startProactiveAlerts: (token: string) => void;
  stopProactiveAlerts: () => void;
  prepareSession: (symbols?: string[]) => Promise<void>;
  refreshSmartPicks: () => Promise<void>;
}

// ── Welcome message ─────────────────────────────────────────────────

const WELCOME_MSG: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content:
    "Welcome to your Trade Co-Planner. Select a stock from the watchlist or ask me anything about your trading.\n\nI can help you:\n• Evaluate trade setups with R:R analysis\n• Check regime suitability for strategies\n• Review your behavioral patterns\n• Plan entries, exits, and position sizing",
  timestamp: new Date(),
};

// ── Store ───────────────────────────────────────────────────────────

export const useCoPlannerStore = create<CoPlannerState>((set, get) => ({
  // Watchlist state
  activeSymbol: null,
  watchlistStocks: [],
  watchlistLoading: false,
  watchlistSearch: "",

  // Chat state
  messages: [WELCOME_MSG],
  sessionId: null,
  chatLoading: false,
  attachments: [],

  // Proactive alerts
  alertWs: null,

  // Session briefing
  briefing: null,
  briefingLoading: false,

  setActiveSymbol: (symbol) => {
    const prev = get().activeSymbol;
    set({ activeSymbol: symbol });
    if (symbol && symbol !== prev) {
      const stock = get().watchlistStocks.find((s) => s.symbol === symbol);
      const contextMsg: ChatMessage = {
        id: `ctx-${Date.now()}`,
        role: "system",
        content: `Context switched to **${symbol}**${
          stock?.matchReason ? ` — ${stock.matchReason}` : ""
        }${stock?.matchScore ? ` (match: ${(stock.matchScore * 100).toFixed(0)}%)` : ""}`,
        timestamp: new Date(),
      };
      set((s) => ({ messages: [...s.messages, contextMsg] }));
      // PROACTIVE PARTNER (merged from the old Partner tab): the partner reads the symbol on the eye and
      // SPEAKS its read into the chat — so picking a symbol feels like a partner tapping your shoulder,
      // not opening a scanner. /partner is sync→threadpooled (safe from the async event loop).
      (async () => {
        try {
          const pr = (await intelligenceApi.partner(symbol, { timeframe: "15m" })) as {
            ok?: boolean; action?: string; speak?: string; headline?: string;
          };
          const body = pr?.speak || pr?.headline;
          if (body && get().activeSymbol === symbol) {
            const partnerMsg: ChatMessage = {
              id: `partner-sym-${Date.now()}`,
              role: "assistant",
              content: `✦ **Partner — ${symbol}**${pr.action ? ` · ${pr.action}` : ""}\n\n${body}`,
              timestamp: new Date(),
            };
            set((s) => ({ messages: [...s.messages, partnerMsg] }));
          }
        } catch { /* partner read is best-effort */ }
      })();
    }
  },

  setWatchlistSearch: (q) => set({ watchlistSearch: q }),

  loadWatchlist: async () => {
    set({ watchlistLoading: true });
    try {
      // Get all strategy watchlists
      const lists = await watchlistApi.list("strategy");
      const allStocks: WatchlistStock[] = [];
      const seen = new Set<string>();

      for (const wl of lists) {
        try {
          const items = await watchlistApi.getItems(wl.id, "matched");
          for (const item of items) {
            if (!seen.has(item.symbol) && (item.match_score ?? 0) > 0.3) {
              seen.add(item.symbol);
              allStocks.push({
                id: item.id,
                symbol: item.symbol,
                matchScore: item.match_score ?? 0,
                matchReason: item.match_reason ?? wl.strategy_name ?? "",
                lastPrice: item.last_analysis?.current_price,
                changePct: item.last_analysis?.change_pct,
                status: item.status,
              });
            }
          }
        } catch {
          // individual watchlist may fail
        }
      }

      // NO auto-seed. Smart Picks is opt-in: if the user has no watchlist
      // stocks yet, show empty. The user can explicitly populate Smart Picks
      // via the refresh action (refreshSmartPicks). The system never scans the
      // FNO universe on its own — scanning is driven by the user's strategies
      // and watchlists only.

      // Sort by match score descending
      allStocks.sort((a, b) => b.matchScore - a.matchScore);
      set({ watchlistStocks: allStocks });
    } catch {
      set({ watchlistStocks: [] });
    } finally {
      set({ watchlistLoading: false });
    }
  },

  sendMessage: async (text) => {
    const { activeSymbol, sessionId, attachments } = get();
    const trimmed = text.trim();
    if (!trimmed) return;

    // Add user message
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: trimmed,
      timestamp: new Date(),
      attachments: attachments.map((f) => ({ name: f.name, type: f.type })),
    };
    set((s) => ({
      messages: [...s.messages, userMsg],
      chatLoading: true,
      attachments: [],
    }));

    try {
      let response: ChatMessageResponse;

      // Always use co-plan endpoint — it injects full trader context
      // (today's stats, streak, drawdown, portfolio heat, behavioral patterns)
      // Attachments (chart screenshots, CSVs) are sent as multipart files
      response = await chatApi.coPlan({
        query: trimmed,
        symbol: activeSymbol ?? undefined,
        session_id: sessionId ?? undefined,
        files: attachments.length > 0 ? attachments : undefined,
      });

      const aiMsg: ChatMessage = {
        id: `ai-${Date.now()}`,
        role: "assistant",
        content: response.message,
        insights: response.insights,
        actionItems: (response as any).action_items,
        followUpQuestions: (response as any).follow_up_questions,
        timestamp: new Date(response.timestamp || Date.now()),
      };

      set((s) => ({
        messages: [...s.messages, aiMsg],
        sessionId: response.session_id || s.sessionId,
        chatLoading: false,
      }));
    } catch (err: any) {
      const errMsg: ChatMessage = {
        id: `err-${Date.now()}`,
        role: "assistant",
        content: `Sorry, I couldn't process that. ${err?.message || "Please try again."}`,
        timestamp: new Date(),
      };
      set((s) => ({
        messages: [...s.messages, errMsg],
        chatLoading: false,
      }));
    }
  },

  addAttachment: (file) =>
    set((s) => ({ attachments: [...s.attachments, file] })),

  removeAttachment: (index) =>
    set((s) => ({ attachments: s.attachments.filter((_, i) => i !== index) })),

  clearAttachments: () => set({ attachments: [] }),

  refreshSmartPicks: async () => {
    set({ watchlistLoading: true });
    try {
      await watchlistApi.smartSeed();
      // Reload after seed
      const lists = await watchlistApi.list("strategy");
      const allStocks: WatchlistStock[] = [];
      const seen = new Set<string>();
      for (const wl of lists) {
        try {
          const items = await watchlistApi.getItems(wl.id, "matched");
          for (const item of items) {
            if (!seen.has(item.symbol) && (item.match_score ?? 0) > 0.3) {
              seen.add(item.symbol);
              allStocks.push({
                id: item.id,
                symbol: item.symbol,
                matchScore: item.match_score ?? 0,
                matchReason: item.match_reason ?? wl.strategy_name ?? "",
                lastPrice: item.last_analysis?.current_price,
                changePct: item.last_analysis?.change_pct,
                status: item.status,
              });
            }
          }
        } catch { /* skip */ }
      }
      allStocks.sort((a, b) => b.matchScore - a.matchScore);
      set({ watchlistStocks: allStocks });
    } catch {
      // silent failure — existing stocks remain
    } finally {
      set({ watchlistLoading: false });
    }
  },

  clearChat: () =>
    set({
      messages: [WELCOME_MSG],
      sessionId: null,
      attachments: [],
    }),

  // ── Proactive alerts via WebSocket ─────────────────────────────
  startProactiveAlerts: (token: string) => {
    const { alertWs } = get();
    if (alertWs && alertWs.readyState < WebSocket.CLOSING) return; // already connected

    const ws = new WebSocket(`${WS_BASE}/api/v1/ws/session?token=${token}`);

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data as string);
        const type: string = msg.type ?? "";
        const data: Record<string, unknown> = msg.data ?? {};

        if (type === "SETUP_DETECTED" || type === "setup.detected") {
          const sym      = (data.symbol as string) ?? "?";
          const setup    = (data.setup_type as string) ?? "setup";
          const setupName = (data.setup_name as string) ?? setup;
          const tf       = (data.timeframe as string) ?? "15m";
          const dir      = (data.direction as string) ?? "";
          const score    = (data.quality_score as number) ?? 0;
          const desc     = (data.description as string) ?? "";
          const meta     = (data.metadata as Record<string, unknown>) ?? {};

          // Grade the signal quality
          const grade = score >= 75 ? "A" : score >= 55 ? "B" : "C";

          // Build a rich structured message for breakout/breakdown signals
          let content = "";
          const isBreakout = setup.includes("breakout") || setup.includes("sr_") || setup.includes("compression");

          if (isBreakout && data.entry_zone) {
            const ez = data.entry_zone as Record<string, number>;
            const stop  = (data.suggested_stop as number);
            const t1    = (data.targets as Record<string, number>)?.t1;
            const t2    = (data.targets as Record<string, number>)?.t2;
            const volOk = (meta.volume_confirmed as boolean);
            const atrEx = (meta.atr_expansion as boolean);
            const sr    = meta.regime_success_rate ? `~${Math.round((meta.regime_success_rate as number)*100)}% success` : "";
            const rr    = stop && t1 && ez.high
              ? ((t1 - ez.high) / (ez.high - stop)).toFixed(1)
              : null;

            // Big move: confirmed breakout/breakdown, high quality, volume + ATR both firing
            const isBigMove = (setupName === "breakout" || setupName === "breakdown"
              || setup === "sr_breakout" || setup === "sr_breakdown")
              && score >= 70 && volOk && atrEx;

            const setupLabel = setupName === "approaching_resistance"
              ? "COMPRESSION FORMING"
              : isBigMove
              ? (setupName === "breakdown" || setup === "sr_breakdown" ? "BIG MOVE DOWN" : "BIG MOVE STARTING")
              : setupName === "breakout" || setup === "sr_breakout"
              ? "BREAKOUT CONFIRMED"
              : "BREAKDOWN CONFIRMED";

            const prefix = isBigMove ? "⚡ " : "";

            content = [
              `${prefix}**${sym} — ${setupLabel} [${tf}]** [${grade}]`,
              desc ? `${desc}` : "",
              `Entry ₹${ez.low?.toFixed(1)}-${ez.high?.toFixed(1)}  |  Stop ₹${stop?.toFixed(1)}  |  T1 ₹${t1?.toFixed(1)}  |  T2 ₹${t2?.toFixed(1)}`,
              rr ? `R:R ${rr}x` : "",
              [volOk ? "volume OK" : "low volume", atrEx ? "ATR expanding" : "", sr].filter(Boolean).join("  •  "),
            ].filter(Boolean).join("\n");
          } else {
            content = `**${sym}** — ${setupName.replace(/_/g, " ")} [${tf}] [${grade}]${dir ? ` (${dir})` : ""}\n${desc || "Setup detected — review before entry."}`;
          }

          const alertMsg: ChatMessage = {
            id: `alert-${Date.now()}`,
            role: "system",
            content,
            timestamp: new Date(),
          };
          set((s) => ({ messages: [...s.messages, alertMsg] }));
        }

        if (type === "STRATEGY_RELOADED" || type === "strategy.reloaded") {
          const name = (data.strategy_name as string) ?? "strategy";
          const reloadMsg: ChatMessage = {
            id: `reload-${Date.now()}`,
            role: "system",
            content: `🔄 **Strategy Updated** — "${name}" is now active in the scanner and intelligence engine.`,
            timestamp: new Date(),
          };
          set((s) => ({ messages: [...s.messages, reloadMsg] }));
        }

        if (type === "COGNITIVE_ALERT" || type === "cognitive.alert") {
          // Proactive PARTNER alert — a money move building on the watchlist, or an open position
          // whose story is changing. Carries the full partner speak (coach + action) → post as chat.
          if (data.source === "partner") {
            const urgency = (data.urgency as string) ?? "medium";
            const message = (data.message as string) ?? (data.headline as string) ?? "";
            if (message) {
              const icon = urgency === "high" ? "🚨" : "🔔";
              const partnerMsg: ChatMessage = {
                id: `partner-${Date.now()}`,
                role: "assistant",
                content: `${icon} **Proactive partner**\n\n${message}`,
                timestamp: new Date(),
              };
              set((s) => ({ messages: [...s.messages, partnerMsg] }));
            }
          }
        }
      } catch {
        // non-JSON frame — ignore
      }
    };

    ws.onclose = () => set({ alertWs: null });
    ws.onerror = () => ws.close();

    set({ alertWs: ws });
  },

  stopProactiveAlerts: () => {
    const { alertWs } = get();
    if (alertWs) {
      alertWs.close();
      set({ alertWs: null });
    }
  },

  // ── Session briefing ───────────────────────────────────────────
  prepareSession: async (symbols?: string[]) => {
    set({ briefingLoading: true });

    // Inject a "thinking" system message
    const thinkingMsg: ChatMessage = {
      id: `briefing-thinking-${Date.now()}`,
      role: "system",
      content: "Preparing your session battle plan — evaluating watchlist, strategies, and market conditions...",
      timestamp: new Date(),
    };
    set((s) => ({ messages: [...s.messages, thinkingMsg] }));

    try {
      const payload = symbols && symbols.length > 0 ? { symbols } : { symbols: [] };
      // WorkflowCopilot returns top_opportunities with {symbol, grade, strategy_name,
      // conditions_met, conditions_total, regime, regime_aligned, readiness_score, estimated_r}
      const report = await chatApi.sessionBriefing(payload) as any;
      set({ briefing: report as SessionBriefingReport });

      // Build opportunity lines from WorkflowCopilot shape
      const oppLines = (report.top_opportunities ?? [])
        .slice(0, 5)
        .map((o: any) => {
          const rStr = o.estimated_r ? `R≈${o.estimated_r.toFixed(1)}x` : "R unknown";
          const condStr = o.conditions_total > 0
            ? `${o.conditions_met}/${o.conditions_total} conds`
            : "";
          const alignStr = o.regime_aligned ? "regime OK" : "regime mismatch";
          return `  • **${o.symbol}** [${o.grade}] — ${o.strategy_name ?? "no strategy"}, ${condStr}, ${rStr}, ${alignStr}`;
        })
        .join("\n");

      const summary = [
        "Session Battle Plan",
        (report.behavioral_summary && report.behavioral_summary !== "No significant behavioral flags")
          ? `\nBehavioral: ${report.behavioral_summary}` : "",
        report.top_opportunities?.length > 0
          ? `\nTop Opportunities:\n${oppLines}`
          : "\nNo qualifying setups detected.",
        report.battle_plan ? `\n---\n${report.battle_plan}` : "",
      ]
        .filter(Boolean)
        .join("\n");

      const briefingMsg: ChatMessage = {
        id: `briefing-${Date.now()}`,
        role: "assistant",
        content: summary,
        timestamp: new Date(),
      };
      // Replace the "thinking" message with the actual briefing
      set((s) => ({
        messages: [
          ...s.messages.filter((m) => m.id !== thinkingMsg.id),
          briefingMsg,
        ],
        briefingLoading: false,
      }));
    } catch (err: unknown) {
      const errMsg: ChatMessage = {
        id: `briefing-err-${Date.now()}`,
        role: "system",
        content: `Session briefing failed. ${err instanceof Error ? err.message : "Please try again."}`,
        timestamp: new Date(),
      };
      set((s) => ({
        messages: [
          ...s.messages.filter((m) => m.id !== thinkingMsg.id),
          errMsg,
        ],
        briefingLoading: false,
      }));
    }
  },
}));
