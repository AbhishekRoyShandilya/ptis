// src/components/trade/FloatingTradeButton.tsx
export function FloatingTradeButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="fixed bottom-6 right-6 h-14 w-14 rounded-full ai-gradient shadow-glow-purple text-white text-2xl font-light hover:opacity-90 active:scale-95 transition-all z-40 flex items-center justify-center"
      title="Log Trade (⌘T)"
    >
      +
    </button>
  );
}
