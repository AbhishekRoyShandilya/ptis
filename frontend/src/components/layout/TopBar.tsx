// src/components/layout/TopBar.tsx
import { useState, useEffect, useRef } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { tokenStore } from "@/services/api";

const NAV_ITEMS = [
  { path: "/", label: "Dashboard", icon: "⬡" },
  { path: "/trades", label: "Trades", icon: "⟡" },
  { path: "/journal", label: "Journal", icon: "◇" },
  { path: "/strategies", label: "Strategies", icon: "⟐" },
  { path: "/intelligence", label: "Intelligence", icon: "💡" },
  { path: "/analytics", label: "Analytics", icon: "◈" },
  { path: "/avt", label: "AVT", icon: "🤖" },
  { path: "/watchlist", label: "Watchlist", icon: "◫" },
  { path: "/system", label: "System", icon: "⚙" },
];

interface TopBarProps {
  disciplineScore?: number | null;
}

export function TopBar({ disciplineScore }: TopBarProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const [isScrolled, setIsScrolled] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const user = tokenStore.getUser();
  const initial = user?.full_name?.[0]?.toUpperCase() || user?.username?.[0]?.toUpperCase() || "U";

  // Close menu on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setUserMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  useEffect(() => {
    const handler = () => setIsScrolled(window.scrollY > 4);
    window.addEventListener("scroll", handler);
    return () => window.removeEventListener("scroll", handler);
  }, []);

  // No fake fallback — a fresh user with no trades has no discipline score yet.
  const hasDisc = disciplineScore != null;
  const discScore = disciplineScore ?? 0;
  const discPct = hasDisc ? (discScore / 10) * 100 : 0;
  const discColor = !hasDisc
    ? "#64748B"
    : discScore >= 7 ? "#10B981" : discScore >= 5 ? "#F59E0B" : "#EF4444";

  return (
    <header
      className={`h-14 flex items-center justify-between px-4 shrink-0 z-50 border-b border-[#2A3050] transition-all duration-200 ${
        isScrolled ? "bg-[#0A0E17]/95 backdrop-blur" : "bg-[#0A0E17]"
      }`}
    >
      {/* Left: Logo */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg ai-gradient flex items-center justify-center shadow-glow-purple">
            <span className="text-white font-bold text-sm">P</span>
          </div>
          <span className="text-sm font-bold text-[#F1F5F9] hidden sm:block tracking-wide">
            PTIS
          </span>
        </div>

        <div className="h-5 w-px bg-[#2A3050]" />

        {/* Nav */}
        <nav className="hidden md:flex items-center gap-0.5">
          {NAV_ITEMS.map((item) => {
            const active = location.pathname === item.path ||
              (item.path !== "/" && location.pathname.startsWith(item.path));
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-150 flex items-center gap-1.5 ${
                  active
                    ? "bg-[#8B5CF6]/15 text-[#A78BFA]"
                    : "text-[#64748B] hover:text-[#94A3B8] hover:bg-[#1A2035]"
                }`}
              >
                <span className={active ? "text-[#8B5CF6]" : ""}>{item.icon}</span>
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Right: Discipline + Avatar */}
      <div className="flex items-center gap-2">
        {/* Discipline Score */}
        <div className="hidden sm:flex items-center gap-2 bg-[#1A2035] rounded-lg px-3 py-1.5">
          <span className="text-[10px] text-[#64748B] uppercase tracking-wide">Discipline</span>
          <div className="w-14 h-1.5 bg-[#0A0E17] rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{ width: `${discPct}%`, backgroundColor: discColor }}
            />
          </div>
          <span
            className="text-sm font-bold font-num"
            style={{ color: discColor }}
          >
            {hasDisc ? discScore.toFixed(1) : "—"}
          </span>
        </div>

        {/* User menu */}
        <div className="relative" ref={menuRef}>
          <button
            onClick={() => setUserMenuOpen((v) => !v)}
            className="h-8 w-8 rounded-full bg-[#8B5CF6]/20 border border-[#8B5CF6]/30 flex items-center justify-center cursor-pointer hover:bg-[#8B5CF6]/30 transition-colors"
          >
            <span className="text-xs font-semibold text-[#A78BFA]">{initial}</span>
          </button>

          {userMenuOpen && (
            <div className="absolute right-0 top-10 w-48 bg-[#1A2035] border border-[#2A3050] rounded-lg shadow-xl z-50 py-1">
              {/* User info */}
              <div className="px-3 py-2 border-b border-[#2A3050]">
                <div className="text-xs font-medium text-[#F1F5F9]">{user?.full_name || user?.username || "User"}</div>
                <div className="text-[10px] text-[#64748B] truncate">{user?.email || ""}</div>
              </div>

              <Link to="/profile" onClick={() => setUserMenuOpen(false)}
                className="block px-3 py-2 text-xs text-[#94A3B8] hover:text-[#F1F5F9] hover:bg-[#0F1629] transition-colors">
                Profile
              </Link>
              <Link to="/settings" onClick={() => setUserMenuOpen(false)}
                className="block px-3 py-2 text-xs text-[#94A3B8] hover:text-[#F1F5F9] hover:bg-[#0F1629] transition-colors">
                Settings
              </Link>

              <div className="border-t border-[#2A3050] mt-1 pt-1">
                <button
                  onClick={() => {
                    tokenStore.clear();
                    setUserMenuOpen(false);
                    navigate("/login");
                  }}
                  className="block w-full text-left px-3 py-2 text-xs text-[#F87171] hover:bg-[#EF4444]/10 transition-colors"
                >
                  Sign Out
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
