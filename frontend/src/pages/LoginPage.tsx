// src/pages/LoginPage.tsx
import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { authApi, tokenStore } from "@/services/api";

export default function LoginPage() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password) return;

    setLoading(true);
    setError(null);

    try {
      const tokens = await authApi.login(username.trim(), password);
      tokenStore.setTokens(tokens.access_token, tokens.refresh_token);

      // Fetch user profile
      const user = await authApi.me();
      tokenStore.setUser(user);

      // Redirect: onboarding if new user, dashboard otherwise
      if (!user.onboarding_completed) {
        navigate("/onboarding");
      } else {
        navigate("/");
      }
    } catch (e: any) {
      setError(e.message || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0F1A] flex items-center justify-center px-4">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-[#F1F5F9] tracking-tight">PTIS</h1>
          <p className="text-xs text-[#64748B] mt-1">Personal Trading Intelligence System</p>
        </div>

        {/* Card */}
        <div className="bg-[#1A2035] border border-[#2A3050] rounded-lg p-6">
          <h2 className="text-sm font-bold text-[#F1F5F9] mb-4">Sign In</h2>

          {error && (
            <div className="mb-4 text-xs text-[#F87171] bg-[#EF4444]/10 border border-[#EF4444]/30 rounded px-3 py-2">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-[10px] text-[#64748B] uppercase mb-1">Username</label>
              <input type="text" value={username} onChange={(e) => setUsername(e.target.value)}
                className="w-full bg-[#0F1629] border border-[#2A3050] text-[#F1F5F9] text-sm rounded px-3 py-2 focus:outline-none focus:border-[#8B5CF6]"
                placeholder="Enter username" autoFocus autoComplete="username" />
            </div>

            <div>
              <label className="block text-[10px] text-[#64748B] uppercase mb-1">Password</label>
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-[#0F1629] border border-[#2A3050] text-[#F1F5F9] text-sm rounded px-3 py-2 focus:outline-none focus:border-[#8B5CF6]"
                placeholder="Enter password" autoComplete="current-password" />
            </div>

            <button type="submit" disabled={loading || !username.trim() || !password}
              className="w-full bg-[#8B5CF6] hover:bg-[#7C3AED] disabled:bg-[#8B5CF6]/30 text-white text-sm font-medium rounded py-2.5 transition-colors">
              {loading ? "Signing in..." : "Sign In"}
            </button>
          </form>

          <div className="mt-4 text-center">
            <span className="text-xs text-[#64748B]">Don't have an account? </span>
            <Link to="/signup" className="text-xs text-[#8B5CF6] hover:text-[#A78BFA]">Create one</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
