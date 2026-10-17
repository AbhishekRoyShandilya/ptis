// src/App.tsx
import { BrowserRouter, Routes, Route, Navigate, Outlet } from "react-router-dom";
import { lazy, Suspense } from "react";
import { tokenStore } from "@/services/api";
import ErrorBoundary from "@/components/common/ErrorBoundary";
import MainLayout from "@/components/layout/MainLayout";

// Eagerly loaded (small, always needed)
import LoginPage from "@/pages/LoginPage";
import SignupPage from "@/pages/SignupPage";

// Lazy loaded (code-split by route)
const OnboardingPage = lazy(() => import("@/pages/OnboardingPage"));
const DashboardPage = lazy(() => import("@/pages/DashboardPage"));
const TradesPage = lazy(() => import("@/pages/TradesPage"));
const JournalPage = lazy(() => import("@/pages/JournalPage"));
const SettingsPage = lazy(() => import("@/pages/SettingsPage"));
const StrategyDetailPage = lazy(() => import("@/pages/StrategyDetailPage"));
const AVTPage = lazy(() => import("@/pages/AVTPage"));
const StrategiesHubPage = lazy(() => import("@/pages/StrategiesHubPage"));
const IntelligenceHubPage = lazy(() => import("@/pages/IntelligenceHubPage"));
const AnalyticsHubPage = lazy(() => import("@/pages/AnalyticsHubPage"));
const SystemHubPage = lazy(() => import("@/pages/SystemHubPage"));
const WatchlistPage = lazy(() => import("@/pages/WatchlistPage"));

function PageLoader() {
  return (
    <div className="flex items-center justify-center min-h-[50vh]">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
    </div>
  );
}

// Auth guard — redirects to /login if not authenticated
function RequireAuth() {
  if (import.meta.env.VITE_DEV_BYPASS !== "true" && !tokenStore.isLoggedIn()) {
    return <Navigate to="/login" replace />;
  }
  return <Outlet />;
}

// Guest guard — redirects to / if already logged in
function GuestOnly() {
  if (tokenStore.isLoggedIn()) {
    return <Navigate to="/" replace />;
  }
  return <Outlet />;
}

export default function App() {
  return (
    <ErrorBoundary>
    <BrowserRouter>
      <Suspense fallback={<PageLoader />}>
      <Routes>
        {/* Public routes (login/signup) */}
        <Route element={<GuestOnly />}>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignupPage />} />
        </Route>

        {/* Onboarding (auth required, no main layout) */}
        <Route element={<RequireAuth />}>
          <Route path="/onboarding" element={<OnboardingPage />} />
        </Route>

        {/* Protected routes (auth required, main layout) */}
        <Route element={<RequireAuth />}>
          <Route element={<MainLayout />}>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/trades" element={<TradesPage />} />
            {/* Co-plan is now embedded in the Dashboard co-plan tab */}
            <Route path="/coplan" element={<Navigate to="/" replace />} />
            <Route path="/coplan/classic" element={<Navigate to="/" replace />} />
            <Route path="/journal" element={<JournalPage />} />
            <Route path="/strategies" element={<StrategiesHubPage />} />
            <Route path="/strategies/:name" element={<StrategyDetailPage />} />
            <Route path="/intelligence" element={<IntelligenceHubPage />} />
            <Route path="/analytics" element={<AnalyticsHubPage />} />
            <Route path="/avt" element={<AVTPage />} />
            <Route path="/watchlist" element={<WatchlistPage />} />
            <Route path="/system" element={<SystemHubPage />} />
            <Route path="/settings" element={<SettingsPage />} />

            {/* Redirects from old routes to hub tabs */}
            <Route path="/behavior" element={<Navigate to="/analytics?tab=behavior" replace />} />
            <Route path="/profile" element={<Navigate to="/analytics?tab=profile" replace />} />
            <Route path="/regime" element={<Navigate to="/analytics?tab=regime" replace />} />
            <Route path="/coaching" element={<Navigate to="/intelligence?tab=coach" replace />} />
            <Route path="/live-alerts" element={<Navigate to="/intelligence?tab=alerts" replace />} />
            <Route path="/bootstrap" element={<Navigate to="/strategies?tab=teach" replace />} />
            <Route path="/discovery" element={<Navigate to="/strategies?tab=discovery" replace />} />
            <Route path="/learning" element={<Navigate to="/strategies?tab=teach" replace />} />
            <Route path="/shadow" element={<Navigate to="/system?tab=shadow" replace />} />
            <Route path="/ske" element={<Navigate to="/system?tab=ske" replace />} />
            <Route path="/watchlists" element={<Navigate to="/watchlist" replace />} />
          </Route>
        </Route>

        {/* Fallback */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      </Suspense>
    </BrowserRouter>
    </ErrorBoundary>
  );
}
