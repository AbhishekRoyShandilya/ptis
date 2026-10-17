// src/services/api.ts
// Central API client for PTIS backend

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── Token Management ──────────────────────────────────────────────
const TOKEN_KEY = "ptis_access_token";
const REFRESH_KEY = "ptis_refresh_token";
const USER_KEY = "ptis_user";

export const tokenStore = {
  getToken: () => localStorage.getItem(TOKEN_KEY),
  getRefresh: () => localStorage.getItem(REFRESH_KEY),
  getUser: (): AuthUser | null => {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? JSON.parse(raw) : null;
  },
  setTokens: (access: string, refresh: string) => {
    localStorage.setItem(TOKEN_KEY, access);
    localStorage.setItem(REFRESH_KEY, refresh);
  },
  setUser: (user: AuthUser) => {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  },
  clear: () => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(REFRESH_KEY);
    localStorage.removeItem(USER_KEY);
  },
  isLoggedIn: () => !!localStorage.getItem(TOKEN_KEY),
};

export interface AuthUser {
  user_id: string;
  username: string;
  email: string;
  full_name: string;
  is_active: boolean;
  created_at: string;
  onboarding_completed?: boolean;
}

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  // Only the PUBLIC auth endpoints must NOT receive the bearer token.
  // Authenticated auth endpoints (/auth/me, /auth/profile, /auth/logout) still
  // need it — stripping it from those caused signup's me() call to 401.
  const AUTH_PUBLIC_PATHS = [
    "/api/v1/auth/register",
    "/api/v1/auth/login",
    "/api/v1/auth/refresh",
  ];
  const isPublicAuth = AUTH_PUBLIC_PATHS.some((p) => path.startsWith(p));

  // Inject auth token if available (skip only for public auth endpoints)
  const token = tokenStore.getToken();
  if (token && !isPublicAuth) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      ...options,
      headers,
    });
  } catch (networkErr) {
    throw new Error("Network error — check your connection and try again");
  }

  // Auto-refresh on 401 (but not for public auth endpoints — a 401 from
  // login/register is a real failure, not an expired session).
  if (res.status === 401 && !isPublicAuth) {
    const refreshed = await _tryRefresh();
    if (refreshed) {
      headers["Authorization"] = `Bearer ${tokenStore.getToken()}`;
      const retryRes = await fetch(`${BASE_URL}${path}`, { ...options, headers });
      if (!retryRes.ok) {
        const err = await retryRes.json().catch(() => ({ detail: "Request failed" }));
        throw new Error(err.detail || "Request failed");
      }
      return retryRes.json();
    }
    // Refresh failed — clear tokens and redirect
    tokenStore.clear();
    window.location.href = "/login";
    throw new Error("Session expired");
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    const detail = err.detail;
    const msg = Array.isArray(detail)
      ? detail.map((d: any) => d.msg || JSON.stringify(d)).join("; ")
      : (typeof detail === "string" ? detail : "Request failed");
    throw new Error(msg);
  }
  return res.json();
}

async function _tryRefresh(): Promise<boolean> {
  const refreshToken = tokenStore.getRefresh();
  if (!refreshToken) return false;
  try {
    const res = await fetch(`${BASE_URL}/api/v1/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    if (!res.ok) return false;
    const data = await res.json();
    tokenStore.setTokens(data.access_token, data.refresh_token);
    return true;
  } catch {
    return false;
  }
}

// ── Auth API ──────────────────────────────────────────────────────
export const authApi = {
  register: (data: { username: string; email: string; password: string; full_name?: string }) =>
    request<AuthUser>("/api/v1/auth/register", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  login: (username: string, password: string) =>
    request<{ access_token: string; refresh_token: string; token_type: string; expires_in: number }>(
      "/api/v1/auth/login",
      { method: "POST", body: JSON.stringify({ username, password }) },
    ),
  me: () => request<AuthUser>("/api/v1/auth/me"),
  refresh: (refresh_token: string) =>
    request<{ access_token: string; refresh_token: string }>("/api/v1/auth/refresh", {
      method: "POST",
      body: JSON.stringify({ refresh_token }),
    }),
  updateProfile: (data: { onboarding_data: Record<string, unknown>; onboarding_completed?: boolean }) =>
    request<AuthUser>("/api/v1/auth/profile", {
      method: "POST",
      body: JSON.stringify({ onboarding_completed: true, ...data }),
    }),
};

// ── Dashboard ─────────────────────────────────────────────────────
export const dashboardApi = {
  get: (regime?: string) =>
    request<DashboardData>(
      `/api/v1/dashboard${regime ? `?current_regime=${regime}` : ""}`
    ),
};

// ── Trades ────────────────────────────────────────────────────────
export const tradesApi = {
  list: (params?: { status?: string; symbol?: string; limit?: number }) => {
    const q = new URLSearchParams(params as Record<string, string>);
    return request<Trade[]>(`/api/v1/trades?${q}`);
  },
  get: (id: string) => request<Trade>(`/api/v1/trades/${id}`),
  enter: (body: TradeEntryRequest) =>
    request<TradeEntryResponse>("/api/v1/trades", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  close: (id: string, body: TradeCloseRequest) =>
    request(`/api/v1/trades/${id}/close`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  // Import the FULL trade history from a Groww F&O P&L statement (.xlsx/.csv) — the intraday-only API
  // can't provide history, but the tax P&L export has every dated round-trip trade.
  importPnl: async (file: File): Promise<{ imported: number; duplicates: number; skipped: number; errors: number; rows_seen: number }> => {
    const fd = new FormData();
    fd.append("file", file);
    const token = tokenStore.getToken();
    const headers: Record<string, string> = {};
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const res = await fetch(`${BASE_URL}/api/v1/broker/import-pnl`, { method: "POST", headers, body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Import failed" }));
      throw new Error(err.detail || "Import failed");
    }
    return res.json();
  },
};

// ── Analytics ─────────────────────────────────────────────────────
export const analyticsApi = {
  performance: () => request("/api/v1/analytics/performance"),
  strategies: () => request("/api/v1/analytics/strategies"),
  equity: () => request("/api/v1/analytics/equity"),
  session: () => request("/api/v1/analytics/session"),
  streak: () => request("/api/v1/analytics/streak"),
  behavior: () => request("/api/v1/analytics/behavior"),
  decisionQuality: (tradeId: string) =>
    request(`/api/v1/analytics/trades/${tradeId}/decision-quality`),
};

// ── Shadow Trading ─────────────────────────────────────────────────
export const shadowApi = {
  sessions: (status?: string) =>
    request<ShadowSession[]>(
      `/api/v1/shadow/sessions${status ? `?status=${status}` : ""}`
    ),
  trades: (sessionId?: string) =>
    request<ShadowTrade[]>(
      `/api/v1/shadow/trades${sessionId ? `?session_id=${sessionId}` : ""}`
    ),
  summary: () => request("/api/v1/shadow/summary"),
  comparison: () => request("/api/v1/shadow/comparison"),
};

// ── Behavior ──────────────────────────────────────────────────────
export const behaviorApi = {
  baseline: (days = 90) => request(`/api/v1/behavior/baseline?lookback_days=${days}`),
  biases: () => request("/api/v1/behavior/biases"),
  deviations: () => request("/api/v1/behavior/deviations"),
  disciplineScore: () => request("/api/v1/behavior/discipline"),
  improvement: (weeks = 12) => request(`/api/v1/behavior/improvement?weeks=${weeks}`),
  dashboard: () => request("/api/v1/behavior/dashboard"),
  alerts: () => request("/api/v1/behavior/alerts"),
};

// ── Coaching Chat ─────────────────────────────────────────────────
export const chatApi = {
  sendMessage: (body: { message: string; session_id?: string }) =>
    request<ChatMessageResponse>("/api/v1/chat/message", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  evaluateTrade: (body: {
    query: string;
    symbol: string;
    direction?: number;
    timeframe?: string;
    session_id?: string;
  }) =>
    request<ChatMessageResponse>("/api/v1/chat/evaluate-trade", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  coPlan: async (body: {
    query: string;
    symbol?: string;
    session_id?: string;
    personality?: string;
    files?: File[];
  }): Promise<ChatMessageResponse> => {
    const formData = new FormData();
    formData.append("query", body.query);
    if (body.symbol) formData.append("symbol", body.symbol);
    if (body.session_id) formData.append("session_id", body.session_id);
    if (body.personality) formData.append("personality", body.personality);
    if (body.files) {
      for (const file of body.files) {
        formData.append("files", file);
      }
    }
    // No Content-Type header — browser sets multipart boundary automatically
    const token = tokenStore.getToken();
    const headers: Record<string, string> = {};
    if (token) headers["Authorization"] = `Bearer ${token}`;
    // Hard timeout — the partner must NEVER sit on an infinite "..." spinner. 120s covers the heaviest
    // legitimate case (a deep multi-trade analysis: ~20s eye read per trade × 4 = ~80s + LLM); past that
    // we abort with a clear message rather than hang forever (e.g. if a background refresh is contending).
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 120_000);
    let res: Response;
    try {
      res = await fetch(`${BASE_URL}/api/v1/chat/co-plan`, {
        method: "POST",
        headers,
        body: formData,
        signal: controller.signal,
      });
    } catch (e: any) {
      if (e?.name === "AbortError") {
        throw new Error(
          "That took too long to answer — the partner may be busy refreshing market data. Please try again in a moment.",
        );
      }
      throw e;
    } finally {
      clearTimeout(timer);
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Request failed" }));
      throw new Error(err.detail || "Co-plan request failed");
    }
    return res.json();
  },
  sessions: () => request("/api/v1/chat/sessions"),
  history: (sessionId: string) => request(`/api/v1/chat/sessions/${sessionId}/history`),
  briefing: () => request("/api/v1/chat/briefing", { method: "POST", body: JSON.stringify({}) }),
  sessionBriefing: (body: { symbols?: string[] }) =>
    request("/api/v1/chat/session-briefing", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};


// ── Regime ───────────────────────────────────────────────────────
export const regimeApi = {
  current: () => request("/api/v1/regime/current"),
  detect: (symbol = "NIFTY", timeframe = "1d") =>
    request(`/api/v1/regime/detect?symbol=${encodeURIComponent(symbol)}&timeframe=${timeframe}`, {
      method: "POST",
    }),
  history: () => request("/api/v1/regime/history"),
  set: (regime: string) =>
    request("/api/v1/regime/set", {
      method: "POST",
      body: JSON.stringify({ regime }),
    }),
  recommendations: (regime: string) =>
    request(`/api/v1/regime/${regime}/recommendations`),
};

// ── Strategies ───────────────────────────────────────────────────
export const strategiesApi = {
  list: () => request("/api/v1/strategies"),
  performance: (id: string) => request(`/api/v1/analytics/strategies/${id}/performance`),
  get: (name: string) => request(`/api/v1/strategies/${name}`),
  checklist: (name: string) => request(`/api/v1/strategies/${name}/checklist`),
  guidance: (name: string) => request(`/api/v1/strategies/${name}/guidance`),
  // system templates (the system's own strategies — cascade futures/options)
  templates: () => request("/api/v1/strategies/templates"),
  activateTemplate: (id: string) =>
    request(`/api/v1/strategies/templates/${id}/activate`, { method: "POST" }),
  deactivateTemplate: (id: string) =>
    request(`/api/v1/strategies/templates/${id}`, { method: "DELETE" }),
};

// ── Journal ──────────────────────────────────────────────────────
export const journalApi = {
  list: (params?: { limit?: number }) =>
    request(`/api/v1/journal?${new URLSearchParams(params as Record<string, string>)}`),
  create: (body: object) =>
    request("/api/v1/journal", { method: "POST", body: JSON.stringify(body) }),
};

// ── Intelligence / Profile ───────────────────────────────────────
export const intelligenceApi = {
  environment: () => request("/api/v1/intelligence/environment", { method: "POST", body: JSON.stringify({}) }),
  scan: (symbols: string[]) =>
    request("/api/v1/intelligence/scan", {
      method: "POST",
      body: JSON.stringify({ symbols }),
    }),
  watchlistScan: (symbols: string[]) =>
    request("/api/v1/intelligence/scan/watchlist", {
      method: "POST",
      body: JSON.stringify({ symbols }),
    }),
  evaluate: (ticker: string, timeframe = "5m") =>
    request("/api/v1/intelligence/evaluate", {
      method: "POST",
      body: JSON.stringify({
        symbol: ticker,
        query: `Evaluate ${ticker} for a potential trade setup`,
        timeframe,
      }),
    }),
  alerts: () => request("/api/v1/intelligence/alerts"),
  cognitiveAlerts: () => request("/api/v1/intelligence/cognitive-alerts"),
  suitability: (symbol: string) => request(`/api/v1/intelligence/suitability/${symbol}`),
  profile: () => request("/api/v1/intelligence/profile"),
  // ── the BRAIN (cognition service) + the COACH face ──
  cognition: (symbol: string, timeframe = "5m", asOf?: string) => {
    const p = new URLSearchParams({ timeframe });
    if (asOf) p.set("as_of", asOf);
    return request(`/api/v1/intelligence/cognition/${symbol}?${p.toString()}`);
  },
  coach: (
    symbol: string,
    opts?: { timeframe?: string; asOf?: string; direction?: string; tradesToday?: number; pnl?: number },
  ) => {
    const p = new URLSearchParams({ timeframe: opts?.timeframe ?? "5m" });
    if (opts?.asOf) p.set("as_of", opts.asOf);
    if (opts?.direction) p.set("direction", opts.direction);
    if (opts?.tradesToday) p.set("trades_today", String(opts.tradesToday));
    if (opts?.pnl) p.set("pnl", String(opts.pnl));
    return request(`/api/v1/intelligence/coach/${symbol}?${p.toString()}`);
  },
  // the UNIFIED PARTNER (co-planner + coach merged): position-aware action + strategy match
  partner: (
    symbol: string,
    opts?: { timeframe?: string; asOf?: string; position?: string; tradesToday?: number; pnl?: number },
  ) => {
    const p = new URLSearchParams({ timeframe: opts?.timeframe ?? "5m" });
    if (opts?.asOf) p.set("as_of", opts.asOf);
    if (opts?.position) p.set("position", opts.position);
    if (opts?.tradesToday) p.set("trades_today", String(opts.tradesToday));
    if (opts?.pnl) p.set("pnl", String(opts.pnl));
    return request(`/api/v1/intelligence/partner/${symbol}?${p.toString()}`);
  },
  // run one proactive pass → publishes partner alerts onto the live feed (ws/session) AND returns them
  proactive: (symbols: string[], asOf?: string) => {
    const p = new URLSearchParams({ symbols: symbols.join(",") });
    if (asOf) p.set("as_of", asOf);
    return request(`/api/v1/intelligence/proactive?${p.toString()}`);
  },
  // EYE-grounded trade review — what the eye saw when you entered/exited (Learning-Journal core)
  tradeReview: (p: { symbol: string; direction: string; entry: string; exit?: string; entryPrice?: number; exitPrice?: number; timeframe?: string }) => {
    const q = new URLSearchParams({ symbol: p.symbol, direction: p.direction, entry: p.entry });
    if (p.exit) q.set("exit", p.exit);
    if (p.entryPrice) q.set("entry_price", String(p.entryPrice));
    if (p.exitPrice) q.set("exit_price", String(p.exitPrice));
    if (p.timeframe) q.set("timeframe", p.timeframe);
    return request(`/api/v1/intelligence/trade-review?${q.toString()}`);
  },
  // teaching session: NL description (+ pointed examples) → grounded review (signature/gaps/eye/validation/refinements)
  teachBootstrap: (body: { description: string; examples?: { symbol: string; when: string; label?: string; interval?: string }[]; interval?: string }) =>
    request(`/api/v1/intelligence/teach/bootstrap`, { method: "POST", body: JSON.stringify(body) }),
  // chat teaching: route a message (description / example / ask) + ground one example on the eye
  teachExtract: (message: string) =>
    request(`/api/v1/intelligence/teach/extract`, { method: "POST", body: JSON.stringify({ message }) }),
  teachGround: (body: { symbol: string; when: string; label?: string; interval?: string }) =>
    request(`/api/v1/intelligence/teach/ground`, { method: "POST", body: JSON.stringify(body) }),
  // persistent strategy SESSIONS (chat-history-per-strategy = My Strategies)
  teachSessions: () => request(`/api/v1/intelligence/teach/sessions`),
  teachSession: (name: string) => request(`/api/v1/intelligence/teach/session/${encodeURIComponent(name)}`),
  teachReviewSignature: (body: { direction: string; require: string[]; exclude: string[] }) =>
    request(`/api/v1/intelligence/teach/review-signature`, { method: "POST", body: JSON.stringify(body) }),
  teachAsk: (body: { question: string; direction: string; require: string[]; exclude: string[] }) =>
    request(`/api/v1/intelligence/teach/ask`, { method: "POST", body: JSON.stringify(body) }),
  teachSaveSession: (body: { name: string; direction: string; require: string[]; exclude: string[]; description?: string; messages?: unknown[]; examples?: unknown[]; thread?: string }) =>
    request(`/api/v1/intelligence/teach/save-session`, { method: "POST", body: JSON.stringify(body) }),
  // analysis (money) thread: P&L backtest via avt_trader + grounded Q&A
  analysisBacktest: (body: { name: string; direction: string; require: string[]; exclude: string[]; interval?: string; max_sessions?: number }) =>
    request(`/api/v1/intelligence/analysis/backtest`, { method: "POST", body: JSON.stringify(body) }),
  analysisStatus: (name: string) => request(`/api/v1/intelligence/analysis/backtest/status/${encodeURIComponent(name)}`),
  analysisDiscoveries: (name: string) => request(`/api/v1/intelligence/analysis/discoveries/${encodeURIComponent(name)}`),
  analysisHistory: (name: string) => request(`/api/v1/intelligence/analysis/history/${encodeURIComponent(name)}`),
  analysisVerdict: (name: string) => request(`/api/v1/intelligence/analysis/verdict/${encodeURIComponent(name)}`),
  analysisApplyRefinement: (body: { name: string; op: string; tags: string[]; reason: string }) =>
    request(`/api/v1/intelligence/analysis/apply-refinement`, { method: "POST", body: JSON.stringify(body) }),
  analysisAsk: (body: { question: string; name: string; direction: string; require: string[]; exclude: string[] }) =>
    request(`/api/v1/intelligence/analysis/ask`, { method: "POST", body: JSON.stringify(body) }),
  // taught strategies (signatures over the eye's vocabulary) + their validated stats
  strategies: () => request(`/api/v1/intelligence/strategies`),
  teachStrategy: (body: { name: string; direction: string; require: string[]; exclude: string[] }) =>
    request(`/api/v1/intelligence/strategies`, { method: "POST", body: JSON.stringify(body) }),
  forgetStrategy: (name: string) =>
    request(`/api/v1/intelligence/strategies/${encodeURIComponent(name)}`, { method: "DELETE" }),
};

export const profileApi = {
  dna: () => request("/api/v1/profile/dna"),
};

// ── Pattern Discovery ───────────────────────────────────────────
export const discoveryApi = {
  discover: (input: string, options?: { similarity_threshold?: number; lookback_years?: number; scope?: string }) =>
    request<DiscoveryReport>("/api/v1/discovery/discover", {
      method: "POST",
      body: JSON.stringify({ input, ...options }),
    }),
  list: () => request<{ discoveries: DiscoverySummary[] }>("/api/v1/discovery/"),
  get: (id: string) => request<DiscoveryReport>(`/api/v1/discovery/${id}`),
  accept: (id: string, notes?: string) =>
    request(`/api/v1/discovery/${id}/accept`, {
      method: "POST",
      body: JSON.stringify({ notes: notes || "" }),
    }),
  reject: (id: string, reason?: string) =>
    request(`/api/v1/discovery/${id}/reject`, {
      method: "POST",
      body: JSON.stringify({ reason: reason || "" }),
    }),
  parse: (input: string) =>
    request("/api/v1/discovery/parse", {
      method: "POST",
      body: JSON.stringify({ input }),
    }),
};

// ── Continuous Research (autonomous discovery loop, user-initiated) ──
//
// Workflow:
//   1. researchApi.startContinuousResearch(...) → returns task immediately
//   2. Poll researchApi.getTask(id) for progress
//   3. Poll researchApi.listDrafts({task_id, status: "pending_review"}) for new drafts
//   4. researchApi.sendDraftToAvt(draft_id) → strategy YAML written, AVT picks up

export interface ResearchTask {
  id: string;
  trader_id: string | null;
  theme: string | null;
  scope: string;
  symbols: string[];
  pattern_types: string[];
  lookback_years: number;
  similarity_threshold: number;
  min_sample_size: number;
  min_win_rate: number;
  min_avg_r: number;
  status: "pending" | "running" | "stopped" | "completed" | "failed";
  candidates_planned: number;
  candidates_completed: number;
  candidates_validated: number;
  current_candidate: string | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  stopped_at: string | null;
}

export interface DiscoveryDraft {
  id: string;
  task_id: string | null;
  symbol: string;
  pattern_type: string;
  sample_size: number;
  win_rate: number;
  avg_r: number;
  profit_factor: number | null;
  best_regime: string | null;
  best_time_slot: string | null;
  htf_alignment: string | null;
  optimal_target_r: number | null;
  reversal_cliff_r: number | null;
  strategy_name: string;
  status: "pending_review" | "sent_to_avt" | "discarded";
  sent_to_avt_at: string | null;
  avt_strategy_id: string | null;
  yaml_path: string | null;
  user_note: string | null;
  created_at: string;
  updated_at: string;
}

export interface DiscoveryDraftDetail extends DiscoveryDraft {
  strategy_yaml: string;
  strategy_dna: Record<string, any>;
}

export interface StartContinuousResearchOptions {
  scope: string;
  theme?: string;
  pattern_types?: string[];
  lookback_years?: number;
  similarity_threshold?: number;
  max_candidates?: number;
  min_sample_size?: number;
  min_win_rate?: number;
  min_avg_r?: number;
}

export const researchApi = {
  startContinuousResearch: (opts: StartContinuousResearchOptions) =>
    request<ResearchTask>("/api/v1/research/discover-continuous", {
      method: "POST",
      body: JSON.stringify(opts),
    }),

  listTasks: (status?: string, limit = 20) => {
    const q = new URLSearchParams();
    if (status) q.set("status", status);
    q.set("limit", String(limit));
    return request<ResearchTask[]>(`/api/v1/research/tasks?${q}`);
  },

  getTask: (id: string) => request<ResearchTask>(`/api/v1/research/tasks/${id}`),

  stopTask: (id: string) =>
    request<ResearchTask>(`/api/v1/research/tasks/${id}/stop`, { method: "POST" }),

  listDrafts: (opts?: { task_id?: string; status?: string; limit?: number }) => {
    const q = new URLSearchParams();
    if (opts?.task_id) q.set("task_id", opts.task_id);
    if (opts?.status) q.set("status", opts.status);
    if (opts?.limit) q.set("limit", String(opts.limit));
    return request<DiscoveryDraft[]>(`/api/v1/research/drafts?${q}`);
  },

  getDraft: (id: string) => request<DiscoveryDraftDetail>(`/api/v1/research/drafts/${id}`),

  sendDraftToAvt: (id: string, note?: string) =>
    request<DiscoveryDraftDetail>(`/api/v1/research/drafts/${id}/send-to-avt`, {
      method: "POST",
      body: JSON.stringify({ note: note || "" }),
    }),

  discardDraft: (id: string, note?: string) =>
    request<DiscoveryDraft>(`/api/v1/research/drafts/${id}/discard`, {
      method: "POST",
      body: JSON.stringify({ note: note || "" }),
    }),
};

// ── Behavioral Agent (autonomous trade-replay analyst) ──
//
// Workflow:
//   1. behavioralAgentApi.run(...) → returns task_id
//   2. Poll behavioralAgentApi.getStatus(taskId) for completion
//   3. behavioralAgentApi.getProfile() → see synthesized profile
//   4. behavioralAgentApi.getInsights() → top 1-3 actionable insights
//   5. behavioralAgentApi.getReplay(tradeId) → drill-down per trade

export interface BARunResponse {
  task_id: string;
  status: string;
  message: string;
  trader_id: string;
  started_at: string;
}

export interface BATaskStatus {
  task_id: string;
  status: "pending" | "running" | "completed" | "failed";
  trader_id: string;
  started_at: string | null;
  finished_at: string | null;
  duration_seconds: number | null;
  trades_processed: number | null;
  replays_created: number | null;
  matches_attributed: number | null;
  causes_linked: number | null;
  biggest_issue_pattern: string | null;
  error: string | null;
}

export interface BAStrengthOrWeakness {
  pattern: string;
  frequency: number;
  frequency_pct: number;
  avg_severity?: number;
  example?: string;
}

export interface BABiggestIssue {
  pattern: string;
  frequency: number;
  frequency_pct: number;
  loss_count: number;
  sample_size: number;
  impact_text: string;
}

export interface BAProfile {
  trader_id: string;
  strengths: BAStrengthOrWeakness[];
  weaknesses: BAStrengthOrWeakness[];
  biggest_issue: BABiggestIssue | null;
  best_strategy: string | null;
  worst_pattern: string | null;
  discipline_score: number | null;
  causal_summary: { cause: string; frequency: number; frequency_pct: number }[];
  archetype: string | null;
  trades_analyzed_count: number;
  last_analyzed_at: string | null;
}

export interface BAInsight {
  rank: number;
  severity: "critical" | "warning" | "info";
  pattern: string;
  title: string;
  explanation: string;
  action: string;
  sample_size: number;
  frequency_pct: number;
}

export interface BAReplay {
  trade_id: string;
  symbol: string;
  tier: string;
  entry_time: string;
  exit_time: string | null;
  regime_at_entry: string | null;
  regime_at_exit: string | null;
  trend_alignment: string | null;
  peak_r: number | null;
  final_r: number | null;
  early_exit_severity: number | null;
  move_pct_complete_at_entry: number | null;
  rsi_at_entry: number | null;
  payload: Record<string, any>;
  causal_sentence: string | null;
}

export const behavioralAgentApi = {
  run: (sinceDays = 90, forceReplay = false) =>
    request<BARunResponse>("/api/v1/behavioral-agent/run", {
      method: "POST",
      body: JSON.stringify({ since_days: sinceDays, force_replay: forceReplay }),
    }),

  getStatus: (taskId: string) =>
    request<BATaskStatus>(`/api/v1/behavioral-agent/status/${taskId}`),

  getProfile: () =>
    request<BAProfile>("/api/v1/behavioral-agent/profile"),

  getInsights: (maxItems = 3) =>
    request<BAInsight[]>(`/api/v1/behavioral-agent/insights?max_items=${maxItems}`),

  getReplay: (tradeId: string) =>
    request<BAReplay>(`/api/v1/behavioral-agent/replay/${tradeId}`),
};

export interface DiscoverySummary {
  discovery_id: string;
  status: string;
  query_symbol: string;
  query_pattern: string;
  total_matches: number;
  win_rate: number;
  created_at: string;
}

export interface DiscoveryReport {
  discovery_id: string;
  status: string;
  created_at: string;
  query: {
    symbol: string;
    timeframe: string;
    event_time: string | null;
    event_date: string | null;
    pattern_type: string;
    description: string;
    scope: string;
    similarity_threshold: number;
  } | null;
  fingerprint: { vector: number[]; labels: string[] } | null;
  total_matches: number;
  overall_win_rate: number;
  overall_avg_r: number;
  overall_profit_factor: number;
  median_hold_minutes: number;
  avg_mae: number;
  top_stocks: Array<{ symbol: string; instances: number; win_rate: number; avg_r: number }>;
  correlation: {
    regime_breakdown: Array<{
      regime: string; count: number; win_rate: number; avg_r: number;
      profit_factor: number; why_works: string; why_fails: string;
    }>;
    time_breakdown: Array<{
      slot: string; count: number; win_rate: number; avg_r: number; label: string;
    }>;
    best_regime: string;
    worst_regime: string;
    best_time_window: string;
  } | null;
  generated_strategy: {
    strategy_name: string;
    display_name: string;
    short_code: string;
    category: string;
    yaml_content: string;
    confidence: number;
    llm_reasoning: string;
  } | null;
  accepted: boolean | null;
  error: string | null;
};

// ─── Types ──────────────────────────────────────────────────────
export interface DashboardData {
  timestamp: string;
  current_regime: string | null;
  today: {
    total_trades: number;
    open_trades: number;
    closed_trades: number;
    pnl: number;
    winners: number;
    losers: number;
    win_rate: number;
    avg_discipline: number | null;
  };
  week: { pnl: number; trades: number };
  open_trades: Array<{
    id: string;
    symbol: string;
    direction: string;
    entry_price: number;
    strategy: string;
    entry_time: string;
  }>;
  open_shadows: number;
  alerts: {
    consecutive_losses: number;
    critical_alerts: number;
    warnings: number;
    insights: AgentInsight[];
  };
  regime_recommendations: Array<{
    strategy: string;
    display_name: string;
    fitness: number;
    sizing: string;
  }>;
}

export interface Trade {
  id: string;
  symbol: string;
  direction: "long" | "short";
  status: "open" | "closed" | "cancelled";
  entry_price: number;
  exit_price: number | null;
  entry_time: string;
  exit_time: string | null;
  position_size: number;
  stop_loss: number | null;
  take_profit: number | null;
  pnl: number | null;          // NET P&L (after charges)
  pnl_pct: number | null;
  r_multiple: number | null;
  commission: number | null;   // brokerage + taxes (estimated)
  risk_amount: number | null;
  session_type: string | null;
  market_regime: string | null;
  declared_strategy_id: string | null;
  emotion_pre: string | null;
  emotion_post: string | null;
  conviction_score: number | null;
  notes: string | null;
  source: string | null;
  resolved_strategy: string | null;
  discipline_score: number | null;
  confidence_score: number | null;
  override_count: number;
}

export interface TradeEntryRequest {
  symbol: string;
  direction: string;
  entry_price: number;
  position_size: number;
  entry_time?: string;
  stop_loss?: number;
  take_profit?: number;
  declared_strategy_id?: string;
  emotion_pre?: string;
  conviction_score?: number;
  notes?: string;
  account_balance?: number;
  source?: string;
}

export interface TradeEntryResponse {
  success: boolean;
  trade_id: string;
  attribution: object | null;
  shadow_trade_ids: string[];
  insights: AgentInsight[];
  warnings: string[];
  errors: string[];
}

export interface TradeCloseRequest {
  exit_price: number;
  exit_time?: string;
  emotion_post?: string;
  exit_notes?: string;
}

export interface AgentInsight {
  agent: string;
  type: string;
  severity: "critical" | "warning" | "info" | "positive";
  title: string;
  message: string;
  strategy: string | null;
  action: string | null;
  trade_ids: string[];
  timestamp: string;
}

export interface ShadowSession {
  id: string;
  session_type: string;
  status: string;
  started_at: string;
  trades: number;
}

export interface ShadowTrade {
  id: string;
  session_id: string;
  type: string;
  symbol: string;
  entry: number;
  exit: number | null;
  stop: number;
  status: string;
  pnl: number | null;
  r_multiple: number | null;
}

export interface ChatMessageResponse {
  session_id: string;
  message: string;
  insights: AgentInsight[];
  timestamp: string;
}

// ── Strategy Learning ─────────────────────────────────────────────
export interface LearningProfile {
  id: string;
  name: string;
  short_code: string;
  status: string;
  narrative: string | null;
  entry_criteria: Record<string, unknown>[];
  exit_rules: Record<string, unknown>;
  risk_parameters: Record<string, unknown>;
  regime_preferences: Record<string, unknown>;
  confidence_scores: Record<string, number>;
  examples_count: number;
  sessions_count: number;
  tags: string[];
}

export interface LearningSession {
  id: string;
  strategy_profile_id: string;
  status: string;
  session_type: string;
  examples_count: number;
  notes: string | null;
  ai_summary: string | null;
}

export interface LearningExample {
  id: string;
  screenshot_path: string;
  vision_analysis: Record<string, unknown>;
  user_annotations: Record<string, unknown>;
  ai_interpretation: string | null;
  is_positive_example: boolean;
  market_context: Record<string, unknown>;
}

export interface CoplanResult {
  setup_match_score: number;
  setup_detected: boolean;
  entry_recommendation: string;
  stop_recommendation: string;
  target_recommendations: string[];
  risk_assessment: string;
  narrative: string;
  confidence: number;
  warnings: string[];
  similar_examples: { example_id: string; similarity_score: number; ai_interpretation: string }[];
}

export const learningApi = {
  // Profiles
  listProfiles: () => request<LearningProfile[]>("/api/v1/learning/profiles"),
  getProfile: (id: string) => request<LearningProfile>(`/api/v1/learning/profiles/${id}`),
  createProfile: (body: { name: string; short_code: string; tags?: string[] }) =>
    request<LearningProfile>("/api/v1/learning/profiles", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  archiveProfile: (id: string) =>
    request(`/api/v1/learning/profiles/${id}`, { method: "DELETE" }),

  // Sessions
  startSession: (body: { profile_id: string; session_type?: string; notes?: string }) =>
    request<LearningSession>("/api/v1/learning/sessions", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getSession: (id: string) => request<LearningSession>(`/api/v1/learning/sessions/${id}`),
  uploadScreenshot: async (sessionId: string, file: File, context?: string) => {
    const formData = new FormData();
    formData.append("file", file);
    if (context) formData.append("context", context);
    const res = await fetch(`${BASE_URL}/api/v1/learning/sessions/${sessionId}/screenshot`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("Screenshot upload failed");
    return res.json() as Promise<LearningExample>;
  },
  annotate: (sessionId: string, body: { example_id: string; annotations: Record<string, unknown> }) =>
    request<LearningExample>(`/api/v1/learning/sessions/${sessionId}/annotate`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  completeSession: (sessionId: string) =>
    request<LearningSession>(`/api/v1/learning/sessions/${sessionId}/complete`, {
      method: "POST",
    }),
  abandonSession: (sessionId: string) =>
    request(`/api/v1/learning/sessions/${sessionId}`, { method: "DELETE" }),

  // Examples & Analysis
  getExamples: (profileId: string) =>
    request<LearningExample[]>(`/api/v1/learning/profiles/${profileId}/examples`),
  getQuestions: (profileId: string) =>
    request<{ questions: string[] }>(`/api/v1/learning/profiles/${profileId}/questions`),
  analyzeTrade: (profileId: string, body: Record<string, unknown>) =>
    request(`/api/v1/learning/profiles/${profileId}/analyze-trade`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // Co-planning
  coplan: async (profileId: string, file: File, context?: string, symbol?: string, timeframe?: string) => {
    const formData = new FormData();
    formData.append("profile_id", profileId);
    formData.append("file", file);
    if (context) formData.append("context", context);
    if (symbol) formData.append("symbol", symbol);
    if (timeframe) formData.append("timeframe", timeframe);
    const res = await fetch(`${BASE_URL}/api/v1/learning/coplan`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("Co-plan request failed");
    return res.json() as Promise<CoplanResult>;
  },
};


// ── Watchlist ──────────────────────────────────────────────────────
export const watchlistApi = {
  list: (type?: string) =>
    request<any[]>(`/api/v1/watchlist${type ? `?watchlist_type=${type}` : ""}`),

  get: (id: string) =>
    request<any>(`/api/v1/watchlist/${id}`),

  create: (data: { name: string; watchlist_type?: string; description?: string; strategy_profile_id?: string }) =>
    request<any>("/api/v1/watchlist", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  delete: (id: string) =>
    request<any>(`/api/v1/watchlist/${id}`, { method: "DELETE" }),

  getItems: (watchlistId: string, status?: string) =>
    request<any[]>(`/api/v1/watchlist/${watchlistId}/items${status ? `?status=${status}` : ""}`),

  addSymbols: (watchlistId: string, symbols: string[]) =>
    request<any>(`/api/v1/watchlist/${watchlistId}/items`, {
      method: "POST",
      body: JSON.stringify({ symbols }),
    }),

  removeSymbol: (watchlistId: string, symbol: string) =>
    request<any>(`/api/v1/watchlist/${watchlistId}/items/${symbol}`, { method: "DELETE" }),

  scan: (watchlistId: string) =>
    request<any>(`/api/v1/watchlist/${watchlistId}/scan`, { method: "POST" }),

  getStrategyMatches: (watchlistId: string) =>
    request<any>(`/api/v1/watchlist/${watchlistId}/strategy-matches`),

  getInsights: (profileId: string) =>
    request<any>(`/api/v1/watchlist/insights/${profileId}`),

  getPrices: (watchlistId: string) =>
    request<WatchlistPricesResponse>(`/api/v1/watchlist/${watchlistId}/prices`),

  smartSeed: () =>
    request<any>("/api/v1/watchlist/smart-seed", { method: "POST" }),

  bundles: () =>
    request<any[]>("/api/v1/watchlist/bundles"),

  addBundle: (watchlistId: string, bundle: string) =>
    request<any>(`/api/v1/watchlist/${watchlistId}/add-bundle?bundle=${encodeURIComponent(bundle)}`, { method: "POST" }),
};

export interface WatchlistPriceItem {
  symbol: string;
  ltp: number | null;
  change: number | null;
  change_pct: number | null;
  match_score: number | null;
  status: string;
}

export interface WatchlistPricesResponse {
  watchlist_id: string;
  watchlist_name: string;
  source: string | null;
  count: number;
  prices: WatchlistPriceItem[];
}

// ── Strategy Knowledge Engine ────────────────────────────────────
export const skeApi = {
  create: (data: { name: string; short_code: string; description?: string; tags?: string[]; scoring_config?: any; conviction_map?: any; entry_zones?: any[]; exit_rules?: any; risk_parameters?: any }) =>
    request<any>("/api/v1/ske/objects", { method: "POST", body: JSON.stringify(data) }),

  list: (status?: string) =>
    request<any[]>(`/api/v1/ske/objects${status ? `?status=${status}` : ""}`),

  get: (id: string) =>
    request<any>(`/api/v1/ske/objects/${id}`),

  update: (id: string, data: any) =>
    request<any>(`/api/v1/ske/objects/${id}`, { method: "PUT", body: JSON.stringify(data) }),

  archive: (id: string) =>
    request<any>(`/api/v1/ske/objects/${id}`, { method: "DELETE" }),

  build: (id: string) =>
    request<any>(`/api/v1/ske/objects/${id}/build`, { method: "POST" }),

  validate: (id: string) =>
    request<any>(`/api/v1/ske/objects/${id}/validate`, { method: "POST" }),

  variants: (id: string) =>
    request<any[]>(`/api/v1/ske/objects/${id}/variants`),

  signal: (id: string, data: { symbol: string; candles?: any[] }) =>
    request<any>(`/api/v1/ske/objects/${id}/signal`, { method: "POST", body: JSON.stringify(data) }),

  batchSignal: (id: string, symbols: string[]) =>
    request<any>(`/api/v1/ske/objects/${id}/signal/batch`, { method: "POST", body: JSON.stringify({ symbols }) }),

  signals: (status?: string, band?: string) => {
    const params = new URLSearchParams();
    if (status) params.set("status", status);
    if (band) params.set("band", band);
    const qs = params.toString();
    return request<any[]>(`/api/v1/ske/signals${qs ? `?${qs}` : ""}`);
  },

  recordOutcome: (signalId: string, outcome_r: number, status = "taken") =>
    request<any>(`/api/v1/ske/signals/${signalId}/outcome?outcome_r=${outcome_r}&status=${status}`, { method: "PUT" }),

  score: (data: { feature_values: Record<string, number>; scoring_config: any; conviction_map?: any }) =>
    request<any>("/api/v1/ske/score", { method: "POST", body: JSON.stringify(data) }),

  regimeMatch: (id: string) =>
    request<any>(`/api/v1/ske/objects/${id}/regime-match`),
};

// ── Improvement Lab ────────────────────────────────────────────
export const improvementApi = {
  analyze: (strategy_id: string) =>
    request<any>("/api/v1/improvement/analyze", {
      method: "POST",
      body: JSON.stringify({ strategy_id }),
    }),
  report: (strategy_id: string) =>
    request<any>(`/api/v1/improvement/report/${strategy_id}`),
  realTimeCheck: (params: {
    symbol?: string;
    direction?: string;
    market_regime?: string;
    session_type?: string;
    emotion_pre?: string;
    conviction_score?: number;
    context?: Record<string, any>;
  }) =>
    request<any>("/api/v1/improvement/real-time-check", {
      method: "POST",
      body: JSON.stringify(params),
    }),
  experiments: (strategy_id?: string) => {
    const qs = strategy_id ? `?strategy_id=${strategy_id}` : "";
    return request<any>(`/api/v1/improvement/experiments${qs}`);
  },
  experiment: (id: string) =>
    request<any>(`/api/v1/improvement/experiments/${id}`),
  graduate: (id: string) =>
    request<any>(`/api/v1/improvement/experiments/${id}/graduate`, { method: "POST" }),
  reject: (id: string, reason?: string) =>
    request<any>(`/api/v1/improvement/experiments/${id}/reject`, {
      method: "POST",
      body: JSON.stringify({ reason: reason || "" }),
    }),
  summary: (strategy_id?: string) => {
    const qs = strategy_id ? `?strategy_id=${strategy_id}` : "";
    return request<any>(`/api/v1/improvement/summary${qs}`);
  },
};

// ── AVT (Autonomous Virtual Trader) ──────────────────────────────
export interface ReplayJobStatus {
  state: "idle" | "running" | "done" | "error";
  message: string;
  started_at?: string;
  finished_at?: string | null;
  progress?: string[];
  results?: any;
  error?: string | null;
  params?: any;
}

export const avtApi = {
  // Control
  start: (strategies?: string[], virtual_capital?: number) =>
    request<AVTStartResponse>("/api/v1/avt/start", {
      method: "POST",
      body: JSON.stringify({ strategies: strategies || [], virtual_capital }),
    }),
  stop: () =>
    request<AVTStopResponse>("/api/v1/avt/stop", { method: "POST" }),
  status: () =>
    request<AVTStatusResponse>("/api/v1/avt/status"),
  health: () =>
    request<AVTHealth>("/api/v1/avt/health"),

  // Capital sleeves (B.1) — the user's account split (futures / options / equity pools)
  getSleeves: () =>
    request<SleeveConfig>("/api/v1/avt/sleeves"),
  setSleeves: (body: { virtual_capital?: number | null; sleeves: SleeveSpec[] }) =>
    request<{ status: string; message: string; virtual_capital: number | null; sleeves: SleeveSpec[] }>(
      "/api/v1/avt/sleeves", { method: "PUT", body: JSON.stringify(body) }),
  clearSleeves: () =>
    request<{ status: string; message: string }>("/api/v1/avt/sleeves", { method: "DELETE" }),

  // Benchmark replay — build the strategy baseline (edge + per-stock ranking init)
  // by replaying the user's taught strategies over lake history. Runs in the
  // background (minutes); poll replayStatus().
  startReplay: (params?: {
    universe?: string; max_symbols?: number;
    from_date?: string; to_date?: string; strategy?: string;
  }) =>
    request<ReplayJobStatus>("/api/v1/avt/replay/start", {
      method: "POST",
      body: JSON.stringify(params || {}),
    }),
  replayStatus: () =>
    request<ReplayJobStatus>("/api/v1/avt/replay/status"),

  // Trades
  trades: (params?: { status?: string; strategy_id?: string; limit?: number }) => {
    const q = new URLSearchParams();
    if (params?.status) q.set("status", params.status);
    if (params?.strategy_id) q.set("strategy_id", params.strategy_id);
    if (params?.limit) q.set("limit", String(params.limit));
    return request<AVTTradeListResponse>(`/api/v1/avt/trades?${q}`);
  },
  openTrades: (strategy_id?: string) => {
    const qs = strategy_id ? `?strategy_id=${strategy_id}` : "";
    return request<AVTTradeListResponse>(`/api/v1/avt/trades/open${qs}`);
  },
  trade: (id: string) =>
    request<AVTVirtualTrade>(`/api/v1/avt/trades/${id}`),

  // Improvements
  improvements: (params?: { strategy_id?: string; status?: string }) => {
    const q = new URLSearchParams();
    if (params?.strategy_id) q.set("strategy_id", params.strategy_id);
    if (params?.status) q.set("status", params.status);
    return request<AVTImprovementListResponse>(`/api/v1/avt/improvements?${q}`);
  },
  improvement: (id: string) =>
    request<AVTImprovement>(`/api/v1/avt/improvements/${id}`),
  adoptImprovement: (id: string, notes?: string) =>
    request<any>(`/api/v1/avt/improvements/${id}/adopt`, {
      method: "POST",
      body: JSON.stringify({ notes: notes || "" }),
    }),
  improvementPresentation: (id: string) =>
    request<AVTPresentation>(`/api/v1/avt/improvements/${id}/presentation`),

  // Competition
  competition: (strategy_id: string, period_days?: number) => {
    const qs = period_days ? `?period_days=${period_days}` : "";
    return request<AVTCompetitionReport>(`/api/v1/avt/competition/${strategy_id}${qs}`);
  },
  competitionHistory: (strategy_id: string, limit?: number) => {
    const qs = limit ? `?limit=${limit}` : "";
    return request<AVTCompetitionHistory>(`/api/v1/avt/competition/${strategy_id}/history${qs}`);
  },

  // Versions
  versions: (strategy_id: string) =>
    request<AVTVersionHistory>(`/api/v1/avt/versions/${strategy_id}`),

  // Scan logs
  scanLogs: (limit?: number) => {
    const qs = limit ? `?limit=${limit}` : "";
    return request<AVTScanLogList>(`/api/v1/avt/scan-logs${qs}`);
  },

  // Conversation
  chat: (message: string) =>
    request<AVTChatResponse>("/api/v1/avt/chat", {
      method: "POST",
      body: JSON.stringify({ message }),
    }),
  chatHistory: (limit?: number) => {
    const qs = limit ? `?limit=${limit}` : "";
    return request<{ history: { role: string; content: string }[] }>(`/api/v1/avt/chat/history${qs}`);
  },
  proactiveMessage: () =>
    request<{ has_message: boolean; message: string | null }>("/api/v1/avt/chat/proactive"),

  // Observations (behavioral only)
  observations: (params?: { limit?: number }) => {
    const q = new URLSearchParams();
    if (params?.limit) q.set("limit", String(params.limit));
    return request<{ observations: AVTObservation[] }>(`/api/v1/avt/observations?${q}`);
  },
  todayObservations: () =>
    request<AVTDaySummary>("/api/v1/avt/observations/today"),

  // Messages (behavioral triage)
  messages: (limit?: number) => {
    const qs = limit ? `?limit=${limit}` : "";
    return request<{ messages: AVTTriageMessage[]; unread_count: number; critical_count: number }>(`/api/v1/avt/messages${qs}`);
  },
  acknowledgeMessage: (id: string) =>
    request<{ acknowledged: boolean }>(`/api/v1/avt/messages/${id}/ack`, { method: "POST" }),

  // Self-improver status (what AVT is proving)
  provingStatus: () =>
    request<Record<string, any>>("/api/v1/avt/proving"),

  // Strategy Lifecycle
  validateStrategy: (strategy_id: string, trade_history?: any[]) =>
    request<any>(`/api/v1/avt/strategies/${strategy_id}/validate`, {
      method: "POST",
      body: JSON.stringify({ trade_history }),
    }),
  lifecycle: (strategy_id: string) =>
    request<AVTLifecycleInfo>(`/api/v1/avt/strategies/${strategy_id}/lifecycle`),
  lifecycleAll: () =>
    request<AVTLifecycleAllResponse>("/api/v1/avt/strategies/lifecycle/all"),
  startCalibration: (strategy_id: string) =>
    request<any>(`/api/v1/avt/strategies/${strategy_id}/calibration/start`, { method: "POST" }),
  skipCalibration: (strategy_id: string) =>
    request<any>(`/api/v1/avt/strategies/${strategy_id}/calibration/skip`, { method: "POST" }),
  calibrationReport: (strategy_id: string) =>
    request<AVTCalibrationReport>(`/api/v1/avt/strategies/${strategy_id}/calibration/report`),
  annotateCalibration: (strategy_id: string, mismatch_id: string, reason: string) =>
    request<any>(`/api/v1/avt/strategies/${strategy_id}/calibration/annotate`, {
      method: "POST",
      body: JSON.stringify({ mismatch_id, reason }),
    }),
  graduateCalibration: (strategy_id: string) =>
    request<any>(`/api/v1/avt/strategies/${strategy_id}/calibration/graduate`, { method: "POST" }),
  edgeMonitor: (strategy_id: string) =>
    request<AVTEdgeMonitorStatus>(`/api/v1/avt/strategies/${strategy_id}/edge-monitor`),
  resumeStrategy: (strategy_id: string) =>
    request<any>(`/api/v1/avt/strategies/${strategy_id}/resume`, { method: "POST" }),

  // Judgment Learning
  recordSetupDecision: (strategy_id: string, decision: {
    judgment_name: string;
    accepted: boolean;
    candles?: Record<string, any>[];
    compression_start?: number;
    compression_end?: number;
    reason?: string;
  }) =>
    request<{ recorded: boolean; strategy_id: string; judgment_name: string; total_decisions_for_factor: number }>(
      `/api/v1/avt/strategies/${strategy_id}/setup-decision`,
      { method: "POST", body: JSON.stringify(decision) },
    ),
  judgmentStatus: (strategy_id: string) =>
    request<{
      strategy_id: string;
      initialized: boolean;
      learners: string[];
      total_decisions: number;
      explanations: Record<string, string>;
      uncertainty_reductions?: Record<string, number>;
    }>(`/api/v1/avt/strategies/${strategy_id}/judgment-status`),
};

// ── AVT Types ────────────────────────────────────────────────────
export interface CapitalSleeve {
  name: string;
  instruments: string[];
  capital: number;
  equity: number;
  available: number;
  deployed: number;
  open_positions: number;
  max_positions: number;
  drawdown_pct: number;
  size_mode?: string;
  size_value?: number;
}

export interface SleeveSpec {
  name: string;
  capital: number;
  instruments: string[];
  max_positions: number;
  size_mode: "risk_pct" | "lots" | "amount";
  size_value: number;
}

export interface SleeveConfig {
  virtual_capital: number | null;
  sleeves: SleeveSpec[];
  running: boolean;
  live: CapitalSleeve[];
  valid_instruments: string[];
}

interface AVTStatusResponse {
  status: string;
  uptime_seconds: number;
  current_cycle: number;
  open_positions: number;
  total_trades: number;
  portfolio_equity: number;
  portfolio_drawdown_pct: number;
  active_strategies: string[];
  last_scan_at: string | null;
  errors: string[];
  capital_sleeves: CapitalSleeve[];
  partitioned: boolean;
}

interface AVTHealthCycle {
  scanned: number;
  candidates: number;
  ranked: number;
  filtered: number;
  executed: number;
  exits: number;
  duration_ms: number;
}

export interface AVTHealth {
  health: "healthy" | "quiet" | "not_ready";
  reason: string;
  strategies_loaded: number;
  strategies_with_scoring_config: number;
  watchlist_size: number;
  signal_gen_attached: boolean;
  stock_ranker_attached: boolean;
  market_data_source: string;
  market_open: boolean;
  status: string;
  cycle_count: number;
  last_scan_at: string | null;
  open_positions: number;
  last_cycle: AVTHealthCycle | null;
  dominant_skip_reason: string | null;
  recent_errors: string[];
}

interface AVTStartResponse {
  status: string;
  message: string;
  strategies_loaded: number;
}

interface AVTStopResponse {
  status: string;
  message: string;
  open_positions_at_stop: number;
}

interface AVTVirtualTrade {
  id: string;
  strategy_id: string;
  strategy_version: string;
  variant_id: string | null;
  symbol: string;
  direction: string;
  entry_time: string;
  entry_price: number;
  quantity: number;
  stop_price: number;
  conviction_at_entry: number;
  regime_at_entry: Record<string, any>;
  ranking_position: number;
  ranking_score: number;
  max_favorable_r: number;
  exit_time: string | null;
  exit_price: number | null;
  exit_type: string | null;
  exit_reason: string | null;
  pnl_r: number | null;
  pnl_pct: number | null;
  pnl_absolute: number | null;
  active_improvements: string[];
  improvement_exits: Record<string, any>[];
  status: string;
}

interface AVTTradeListResponse {
  trades: AVTVirtualTrade[];
  total: number;
}

interface AVTImprovement {
  id: string;
  strategy_id: string;
  type: string;
  description: string;
  params: Record<string, any>;
  status: string;
  applied_at: string | null;
  applied_at_version: string | null;
  evidence_strength: number;
  performance_before: Record<string, any> | null;
  performance_after: Record<string, any> | null;
  validation_result: Record<string, any>;
  reverted_at: string | null;
  revert_reason: string | null;
}

interface AVTImprovementListResponse {
  improvements: AVTImprovement[];
  total: number;
}

interface AVTPresentation {
  title: string;
  narrative: string;
  summary_stats: Record<string, any>;
  examples: Record<string, any>[];
  user_action_options: string[];
  avt_confidence: number;
}

interface AVTCompetitionReport {
  strategy_name: string;
  period_days: number;
  user_stats: Record<string, any>;
  avt_stats: Record<string, any>;
  performance_gap: number;
  differences: Record<string, any>[];
  proven_improvements: Record<string, any>[];
  overall_narrative: string;
}

interface AVTCompetitionHistory {
  strategy_id: string;
  snapshots: {
    snapshot_date: string;
    period_days: number;
    performance_gap: number | null;
    user_stats: Record<string, any>;
    avt_stats: Record<string, any>;
  }[];
}

interface AVTVersionHistory {
  strategy_id: string;
  versions: {
    id: string;
    version: string;
    started_at: string | null;
    ended_at: string | null;
    active_improvements: string[];
    trade_count: number;
    performance_stats: Record<string, any>;
    notes: string | null;
  }[];
}

interface AVTScanLogList {
  logs: {
    id: string;
    strategy_id: string | null;
    cycle_time: string;
    candidates_found: number;
    trades_opened: number;
    trades_closed: number;
    skipped: Record<string, any>[];
    top_ranked: Record<string, any>[];
    duration_ms: number;
  }[];
  total: number;
}

interface AVTChatResponse {
  response: string;
  intent: string;
  pending_messages: any[];
}

interface AVTObservation {
  id: string;
  type: string;
  source: string;
  tier: string;
  title: string;
  description: string;
  severity: string;
  data: Record<string, any>;
  strategy_id: string | null;
  trade_ids: string[];
  hypothesis_id: string | null;
  timestamp: string;
  acknowledged: boolean;
}

interface AVTDaySummary {
  date: string;
  total_observations: number;
  by_severity: Record<string, number>;
  by_tier: Record<string, number>;
  observations: AVTObservation[];
}

interface AVTTriageMessage {
  id: string;
  type: string;
  priority: number;
  message: string;
  severity: string;
  timestamp: string;
  data: Record<string, any>;
  strategy_id: string | null;
  acknowledged: boolean;
}

// Strategy Lifecycle types
interface AVTLifecycleInfo {
  strategy_id: string;
  state: string;
  grade: string;
  health_score: number;
  allocation_multiplier: number;
  health_issues: { category: string; severity: string; field: string; message: string; suggestion: string }[];
  calibration_alignment: number | null;
  calibration_signals: number;
  calibration_comparisons: number;
  degradation_level: string;
  rolling_expectancy: number | null;
  translation_drift: number | null;
  baseline_expectancy: number | null;
}

interface AVTLifecycleAllResponse {
  strategies: Record<string, {
    state: string;
    grade: string;
    health_score: number;
    allocation_multiplier: number;
    degradation_level: string;
  }>;
  total: number;
}

interface AVTCalibrationReport {
  strategy_id: string;
  alignment_score: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  false_positive_rate: number;
  false_negative_rate: number;
  total_comparisons: number;
  signals_generated: number;
  user_trades: number;
  ready_to_graduate: boolean;
  graduation_blockers: string[];
  graduation_message: string;
  top_mismatch_reasons: string[];
  mismatches: { id: string; type: string; symbol: string; timestamp: string | null; user_annotation: string | null }[];
}

interface AVTEdgeMonitorStatus {
  strategy_id: string;
  level: string;
  action: string;
  rolling_expectancy: number;
  rolling_win_rate: number;
  rolling_trade_count: number;
  drift_score: number;
  baseline_expectancy: number | null;
  baseline_win_rate: number | null;
  total_trades: number;
  message: string;
}


// ══════════════════════════════════════════════════════════════
//  BOOTSTRAP API
// ══════════════════════════════════════════════════════════════

export const bootstrapApi = {
  // Pattern Library
  listPatterns: () =>
    request<{ patterns: BootstrapPatternSummary[] }>("/api/v1/bootstrap/patterns"),
  searchPatterns: (q: string) =>
    request<{ results: BootstrapPatternMatch[] }>(`/api/v1/bootstrap/patterns/search?q=${encodeURIComponent(q)}`),
  getPattern: (id: string) =>
    request<BootstrapPatternDetail>(`/api/v1/bootstrap/patterns/${id}`),

  // Sessions
  start: (input: BootstrapStartInput) =>
    request<BootstrapSessionState>("/api/v1/bootstrap/start", {
      method: "POST",
      body: JSON.stringify(input),
    }),
  getSession: (id: string) =>
    request<BootstrapSessionState>(`/api/v1/bootstrap/sessions/${id}`),
  correct: (id: string, corrections: BootstrapCorrection[]) =>
    request<BootstrapSessionState>(`/api/v1/bootstrap/sessions/${id}/correct`, {
      method: "POST",
      body: JSON.stringify({ corrections }),
    }),
  finalize: (id: string) =>
    request<BootstrapFinalizeResult>(`/api/v1/bootstrap/sessions/${id}/finalize`, {
      method: "POST",
    }),

  // Standalone
  interpret: (description: string, patternIds?: string[]) =>
    request<BootstrapInterpretation>("/api/v1/bootstrap/interpret", {
      method: "POST",
      body: JSON.stringify({ description, pattern_ids: patternIds || [] }),
    }),
};


// ── Bootstrap Types ───────────────────────────────────────────

export interface BootstrapPatternSummary {
  pattern_id: string;
  name: string;
  category: string;
  description: string;
  hard_rules_count: number;
  variants_count: number;
  regime_performance: Record<string, string>;
}

export interface BootstrapPatternMatch {
  pattern_id: string;
  name: string;
  category: string;
  description: string;
  match_score: number;
}

export interface BootstrapPatternDetail {
  pattern_id: string;
  name: string;
  category: string;
  description: string;
  hard_rules: Record<string, any>[];
  soft_scorers: Record<string, any>[];
  context_factors: Record<string, any>[];
  feature_keys: string[];
  variants: { name: string; description: string }[];
  regime_performance: Record<string, string>;
  failure_modes: { name: string; description: string }[];
  uncertainty_factors: Record<string, any>[];
  teaching_questions: string[];
}

export interface BootstrapStartInput {
  description: string;
  selected_pattern_ids?: string[];
  timeframe?: string;
  direction?: string;
  instruments?: string[];
  user_claimed_wr?: number | null;
}

export interface BootstrapCorrection {
  type: string;
  rule_index?: number;
  field?: string;
  new_value?: any;
  text?: string;
  answer?: string;
  rule?: Record<string, any>;
}

export interface BootstrapSessionState {
  session_id: string;
  status: string;
  iteration: number;
  convergence_score: number;
  warnings: string[];
  created_at: string | null;
  updated_at: string | null;
  draft?: {
    strategy_name: string;
    understanding_confidence: number;
    hard_rules_count: number;
    soft_scorers_count: number;
    feature_keys: string[];
    gaps: string[];
    questions: string[];
    warnings: string[];
  };
  backtest?: {
    total_trades: number;
    win_rate: number;
    avg_r: number;
    total_r: number;
    profit_factor: number;
    max_drawdown_r: number;
    avg_synthetic_confidence: number;
    regime_distribution: Record<string, number>;
    warnings: string[];
  };
  warm_start?: {
    scaler_initialized: boolean;
    clusterer_initialized: boolean;
    clusters_found: number;
    regime_checker_initialized: boolean;
    performance_reasonable: boolean;
    recommended_avt_mode: string;
    warnings: string[];
  };
  profile_id?: string;
  yaml_path?: string;
}

export interface BootstrapFinalizeResult {
  profile_id: string;
  yaml_path: string;
  short_code: string;
  strategy_name: string;
}

export interface BootstrapInterpretation {
  strategy_name: string;
  understanding_confidence: number;
  hard_rules: Record<string, any>[];
  soft_scorers: Record<string, any>[];
  feature_keys: string[];
  matched_templates: string[];
  gaps: string[];
  questions: string[];
  warnings: string[];
}


// ── Broker API ───────────────────────────────────────────────

export const brokerApi = {
  status: () =>
    request<BrokerStatusResponse>("/api/v1/broker/status"),
  testAngel: () =>
    request<{ status: string; test_symbol?: string; candles_fetched?: number; last_close?: number; last_date?: string; detail?: string }>(
      "/api/v1/broker/test/angel", { method: "POST" }
    ),
  testUpstox: () =>
    request<{ status: string; test_symbol?: string; ltp?: number; detail?: string }>(
      "/api/v1/broker/test/upstox", { method: "POST" }
    ),
  testGroww: () =>
    request<{ status: string; account?: { equity: number; buying_power: number }; detail?: string }>(
      "/api/v1/broker/test/groww", { method: "POST" }
    ),
  triggerSync: () =>
    request<{ status: string; synced: number; skipped: number; errors: number }>(
      "/api/v1/broker/sync/trigger", { method: "POST" }
    ),
  syncStatus: () =>
    request<SyncStatusResponse>("/api/v1/broker/sync/status"),

  // Portfolio
  positions: () =>
    request<{ broker: string; count: number; positions: BrokerPosition[] }>("/api/v1/broker/positions"),
  orders: (status = "filled", days = 1) =>
    request<{ broker: string; count: number; filter: { status: string; from_date: string }; orders: BrokerOrder[] }>(
      `/api/v1/broker/orders?status=${status}&days=${days}`
    ),
  account: () =>
    request<BrokerAccount>("/api/v1/broker/account"),

  // Market Data
  ltp: (symbol: string) =>
    request<{ symbol: string; ltp: number }>(`/api/v1/broker/ltp/${symbol}`),
  ltpBatch: (symbols: string[]) =>
    request<{ prices: Record<string, number | null>; count: number }>(
      `/api/v1/broker/ltp?symbols=${symbols.join(",")}`
    ),
  quote: (symbol: string) =>
    request<BrokerQuote>(`/api/v1/broker/quote/${symbol}`),
  intraday: (symbol: string, interval = "5m") =>
    request<{ symbol: string; interval: string; count: number; candles: BrokerCandle[] }>(
      `/api/v1/broker/intraday/${symbol}?interval=${interval}`
    ),
  candles: (symbol: string, interval = "1d", days = 30) =>
    request<{ symbol: string; interval: string; from_date: string; to_date: string; count: number; candles: BrokerCandle[] }>(
      `/api/v1/broker/candles/${symbol}?interval=${interval}&days=${days}`
    ),
};

export interface BrokerInfo {
  configured: boolean;
  connected: boolean;
  role: string;
  last_sync?: string | null;
  sync_count?: number;
}

export interface BrokerStatusResponse {
  angel_one: BrokerInfo;
  upstox: BrokerInfo;
  zerodha: BrokerInfo;
  groww: BrokerInfo;
}

export interface BrokerPosition {
  symbol: string;
  exchange: string;
  quantity: number;
  avg_price: number;
  pnl: number;
  side: "long" | "short";
  product: string;
  ltp: number;
}

export interface BrokerOrder {
  symbol: string;
  side: string;
  qty: number;
  price: number;
  filled_at: string | null;
  status: string;
  order_id: string;
  exchange: string;
  product: string;
}

export interface BrokerAccount {
  broker: string;
  equity: number;
  buying_power: number;
  cash: number;
  margin_used: number;
  collateral?: number;
}

export interface BrokerQuote {
  symbol: string;
  ltp: number | null;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  prev_close?: number;
  volume?: number;
  oi?: number;
  bid?: number | null;
  ask?: number | null;
  timestamp?: string;
}

export interface BrokerCandle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface SyncStatusResponse {
  running: boolean;
  last_sync: string | null;
  sync_count: number;
  total_synced: number;
  error_count: number;
  tracked_order_ids: number;
  configured?: boolean;
  detail?: string;
}

// ── Co-Planner Intelligence ─────────────────────────────────────

export interface CoPlanSetupInput {
  symbol: string;
  strategy_name: string;
  timeframe?: string;
  entry_price?: number;
  stop_loss?: number;
  target_price?: number;
  current_volume?: number;
  avg_volume?: number;
  current_regime?: string;
  pattern_score?: number;
  cluster_id?: string | null;
}

export interface CoPlanOpenPosition {
  symbol: string;
  strategy?: string;
  sector?: string;
}

export interface CoPlanBehavioralCheck {
  warnings: { type: string; severity: string; message: string }[];
  risk_score: number;
  max_risk_score: number;
  verdict: string;
  proceed: boolean;
}

export interface CoPlanRankedOpportunity {
  rank: number;
  symbol: string;
  strategy_name: string;
  composite_score: number;
  conviction: {
    conviction_score: number;
    conviction_level: string;
    components: Record<string, number>;
    warnings: { type: string; severity: string; message: string }[];
    sizing_modifier: number;
    verdict: string;
  };
  rr_assessment: {
    expected_rr: number;
    grade: string;
    acceptable: boolean;
    message: string;
    sizing_modifier: number;
  } | null;
  score_breakdown: Record<string, number>;
  action: string;
  sizing: number;
  explanation: string;
}

export interface CoPlanSessionResult {
  behavioral_check: CoPlanBehavioralCheck;
  ranked_opportunities: CoPlanRankedOpportunity[];
  regime_warnings: { symbol: string; strategy: string; regime: string; warning: string; alternatives: any[] }[];
  recommendation: { action: string; primary?: string; message?: string; sizing?: string; reason?: string };
  explanation: string;
  total_scanned: number;
  actionable_count: number;
  computed_at: string;
}

export interface CoPlanEvalResult {
  conviction: {
    score: number;
    level: string;
    components: Record<string, number>;
    warnings: { type: string; severity: string; message: string }[];
    sizing: number;
    verdict: string;
  };
  regime: { win_rate: number; avg_rr: number; trades: number; confidence: string };
  avoidance: { avoid: boolean; reason: string; alternatives: any[] };
  rr: { expected_rr: number; grade: string; acceptable: boolean; message: string } | null;
}

export const coplannerApi = {
  plan: (setups: CoPlanSetupInput[], openPositions?: CoPlanOpenPosition[]) =>
    request<CoPlanSessionResult>("/api/v1/coplanner/plan", {
      method: "POST",
      body: JSON.stringify({ setups, open_positions: openPositions }),
    }),

  evaluate: (setup: CoPlanSetupInput) =>
    request<CoPlanEvalResult>("/api/v1/coplanner/evaluate", {
      method: "POST",
      body: JSON.stringify({ setup }),
    }),

  regimeMatrix: () =>
    request<Record<string, any>>("/api/v1/coplanner/regime-matrix"),

  strategyProfile: (name: string) =>
    request<Record<string, any>>(`/api/v1/coplanner/regime-matrix/${encodeURIComponent(name)}`),

  rebuildMatrix: () =>
    request<Record<string, any>>("/api/v1/coplanner/regime-matrix/rebuild", { method: "POST" }),

  suggestTargets: (entry: number, stopLoss: number) =>
    request<Record<string, any>>("/api/v1/coplanner/rr/suggest-targets", {
      method: "POST",
      body: JSON.stringify({ entry, stop_loss: stopLoss }),
    }),

  rrConfig: () =>
    request<Record<string, any>>("/api/v1/coplanner/rr/config"),

  dataQuality: () =>
    request<Record<string, any>>("/api/v1/coplanner/data-quality"),

  dataAvailability: () =>
    request<Record<string, any>>("/api/v1/coplanner/data-availability"),

  executionProfile: () =>
    request<Record<string, any>>("/api/v1/coplanner/execution-profile"),

  evolutionTimeline: (windowSize = 20) =>
    request<Record<string, any>[]>(`/api/v1/coplanner/evolution-timeline?window_size=${windowSize}`),

  behavioralPatterns: () =>
    request<Record<string, any>>("/api/v1/coplanner/behavioral-patterns"),

  refreshTrustWeights: () =>
    request<Record<string, any>>("/api/v1/coplanner/refresh-trust-weights", { method: "POST" }),

  learnedStrategies: () =>
    request<Record<string, any>>("/api/v1/coplanner/learned-strategies"),

  strategiesForRegime: (regime: string) =>
    request<Record<string, any>>(`/api/v1/coplanner/strategies-for-regime/${encodeURIComponent(regime)}`),
};
