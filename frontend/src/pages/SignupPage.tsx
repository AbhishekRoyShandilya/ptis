// src/pages/SignupPage.tsx
import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { authApi, tokenStore } from "@/services/api";

export default function SignupPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ username: "", email: "", password: "", confirmPassword: "", full_name: "" });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const update = (field: string, value: string) => setForm((f) => ({ ...f, [field]: value }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!form.username.trim() || !form.email.trim() || !form.password) {
      setError("All fields are required");
      return;
    }
    const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
    if (!emailRegex.test(form.email.trim())) {
      setError("Please enter a valid email address");
      return;
    }
    if (form.password.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }
    if (form.password !== form.confirmPassword) {
      setError("Passwords don't match");
      return;
    }

    setLoading(true);
    try {
      // Register
      await authApi.register({
        username: form.username.trim(),
        email: form.email.trim(),
        password: form.password,
        full_name: form.full_name.trim() || undefined,
      });

      // Auto-login after registration
      const tokens = await authApi.login(form.username.trim(), form.password);
      tokenStore.setTokens(tokens.access_token, tokens.refresh_token);

      const user = await authApi.me();
      tokenStore.setUser(user);

      // New user → onboarding
      navigate("/onboarding");
    } catch (e: any) {
      setError(e.message || "Registration failed");
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
          <h2 className="text-sm font-bold text-[#F1F5F9] mb-4">Create Account</h2>

          {error && (
            <div className="mb-4 text-xs text-[#F87171] bg-[#EF4444]/10 border border-[#EF4444]/30 rounded px-3 py-2">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-3">
            <div>
              <label className="block text-[10px] text-[#64748B] uppercase mb-1">Full Name</label>
              <input type="text" value={form.full_name} onChange={(e) => update("full_name", e.target.value)}
                className="w-full bg-[#0F1629] border border-[#2A3050] text-[#F1F5F9] text-sm rounded px-3 py-2 focus:outline-none focus:border-[#8B5CF6]"
                placeholder="Your name" autoFocus />
            </div>

            <div>
              <label className="block text-[10px] text-[#64748B] uppercase mb-1">Username</label>
              <input type="text" value={form.username} onChange={(e) => update("username", e.target.value)}
                className="w-full bg-[#0F1629] border border-[#2A3050] text-[#F1F5F9] text-sm rounded px-3 py-2 focus:outline-none focus:border-[#8B5CF6]"
                placeholder="Choose a username" autoComplete="username" />
            </div>

            <div>
              <label className="block text-[10px] text-[#64748B] uppercase mb-1">Email</label>
              <input type="email" value={form.email} onChange={(e) => update("email", e.target.value)}
                className="w-full bg-[#0F1629] border border-[#2A3050] text-[#F1F5F9] text-sm rounded px-3 py-2 focus:outline-none focus:border-[#8B5CF6]"
                placeholder="you@example.com" autoComplete="email" />
            </div>

            <div>
              <label className="block text-[10px] text-[#64748B] uppercase mb-1">Password</label>
              <input type="password" value={form.password} onChange={(e) => update("password", e.target.value)}
                className="w-full bg-[#0F1629] border border-[#2A3050] text-[#F1F5F9] text-sm rounded px-3 py-2 focus:outline-none focus:border-[#8B5CF6]"
                placeholder="Min 8 characters" autoComplete="new-password" />
            </div>

            <div>
              <label className="block text-[10px] text-[#64748B] uppercase mb-1">Confirm Password</label>
              <input type="password" value={form.confirmPassword} onChange={(e) => update("confirmPassword", e.target.value)}
                className="w-full bg-[#0F1629] border border-[#2A3050] text-[#F1F5F9] text-sm rounded px-3 py-2 focus:outline-none focus:border-[#8B5CF6]"
                placeholder="Repeat password" autoComplete="new-password" />
            </div>

            <button type="submit" disabled={loading}
              className="w-full bg-[#8B5CF6] hover:bg-[#7C3AED] disabled:bg-[#8B5CF6]/30 text-white text-sm font-medium rounded py-2.5 transition-colors mt-1">
              {loading ? "Creating account..." : "Create Account"}
            </button>
          </form>

          <div className="mt-4 text-center">
            <span className="text-xs text-[#64748B]">Already have an account? </span>
            <Link to="/login" className="text-xs text-[#8B5CF6] hover:text-[#A78BFA]">Sign in</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
