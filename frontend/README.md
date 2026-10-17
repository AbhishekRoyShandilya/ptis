# PTIS Frontend v2.0
## Personal Trading Intelligence System — Redesigned Frontend

Completely redesigned frontend adopting **Mirror Web**'s dark aesthetic into a comprehensive, production-ready React + TypeScript + Vite application that wires to every PTIS backend endpoint.

---

## Design System

Adopted directly from Mirror Web:
- **Background**: `#0A0E17` (deep navy)
- **Cards**: `#1A2035` with `#2A3050` borders
- **Accent**: Purple gradient (`#8B5CF6 → #3B82F6`)
- **Profit**: `#10B981` (emerald)
- **Loss**: `#EF4444` (red)
- **Warning**: `#F59E0B` (amber)
- **Numbers**: JetBrains Mono font
- **Glow effects** on active cards

---

## Pages & Backend Wiring

| Page | Route | API Endpoints Used |
|------|-------|-------------------|
| Dashboard | `/` | `GET /api/dashboard/` |
| Trades | `/trades` | `GET/POST /api/trades`, `POST /api/trades/{id}/close` |
| Analytics | `/analytics` | `GET /api/analytics/performance`, `equity`, `strategies` |
| Shadow Trading | `/shadow` | `GET /api/shadow/sessions`, `trades`, `summary`, `comparison` |
| Behavior | `/behavior` | `GET /api/behavior/baseline`, `biases`, `deviations`, `discipline`, `improvement` |
| AI Coach | `/coaching` | `POST /api/chat/message`, `GET /api/chat/sessions` |
| Journal | `/journal` | `GET/POST /api/journal` |
| Regime | `/regime` | `GET /api/regime/current`, `history` |
| Strategies | `/strategies` | `GET /api/strategies` |
| Strategy Detail | `/strategies/:name` | `GET /api/strategies/{name}`, `checklist`, `guidance` |
| Intelligence | `/intelligence` | `POST /api/intelligence/environment`, `scan`, `evaluate`, `alerts`, `suitability`, `profile` |
| Profile | `/profile` | `GET /api/profile/dna` |
| Settings | `/settings` | Configuration UI |
| — | — | **new pages added:** intelligence, profile, strategy detail |
---

## Components Removed from PTIS Original

These PTIS frontend components were placeholder-only and have been rebuilt properly:
- All placeholder `.tsx` files (70 files were `# Auto-generated placeholder`)
- Duplicate/unused component directories
- MUI dependency (replaced with Tailwind + custom components)
- `@emotion` packages (no longer needed)

## New Components Added (from Mirror Web patterns)

- `TopBar` — regime-aware header with discipline score
- `JournalDialog` — quick & voice journal entry modal
- `TradeEntryDialog` — full trade logger with AI insight feedback
- `TradeCloseDialog` — close trade with emotion tracking
- `FloatingTradeButton` — always-visible trade logger (⌘T)
- `AlertQueue` — behavioral deviation alerts banner

---

## Setup

```bash
cd ptis-frontend-v2
npm install
npm run dev
```

### Environment
```env
VITE_API_URL=http://localhost:8000
```

Make sure the PTIS backend is running on port 8000.

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| ⌘J | Open Journal |
| ⌘T | Log Trade |
| Escape | Close any dialog |

---

## Tech Stack

- **React 18** + TypeScript
- **Vite** (fast dev server + build)
- **React Router v6** (client-side routing)
- **Tailwind CSS** (utility-first, Mirror Web design tokens)
- **Recharts** (performance charts)
- **Zustand** (state management)
- **Lucide React** (icons)
- No MUI, No Emotion — clean and fast
