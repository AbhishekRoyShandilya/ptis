/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Mirror Web design tokens adapted for PTIS
        "ptis-bg": "#0A0E17",
        "ptis-surface": "#0F1420",
        "ptis-card": "#1A2035",
        "ptis-card-hover": "#1E2540",
        "ptis-border": "#2A3050",
        "ptis-border-light": "#3A4060",

        // Text hierarchy
        "text-primary": "#F1F5F9",
        "text-secondary": "#CBD5E1",
        "text-muted": "#94A3B8",
        "text-faint": "#64748B",

        // Semantic colors
        profit: "#10B981",
        loss: "#EF4444",
        warning: "#F59E0B",
        "ai-purple": "#8B5CF6",
        "ai-purple-light": "#A78BFA",
        "ai-blue": "#3B82F6",
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      boxShadow: {
        "glow-green": "0 0 20px rgba(16, 185, 129, 0.15)",
        "glow-purple": "0 0 20px rgba(139, 92, 246, 0.2)",
        "glow-amber": "0 0 20px rgba(245, 158, 11, 0.15)",
        "glow-red": "0 0 20px rgba(239, 68, 68, 0.15)",
        card: "0 4px 6px -1px rgba(0, 0, 0, 0.3)",
        "card-hover": "0 8px 15px rgba(0, 0, 0, 0.4)",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-in": "fadeIn 0.3s ease-out",
        "slide-up": "slideUp 0.3s ease-out",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};
