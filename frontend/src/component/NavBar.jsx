import { useTransition } from "react";

const TABS = [
  { id: "prediction", label: "Prediction", icon: "[]", count: null },
  { id: "performance", label: "Performance", icon: "O", count: 84 },
  { id: "analytics", label: "Analytics", icon: "=", count: "1.2k" },
  { id: "comparison", label: "Comparison", icon: "<>", count: 12 },
];

export default function NavBar({
  activeTab = "prediction",
  onTabChange = () => {},
}) {
  const [, startTransition] = useTransition();

  return (
    <nav
      className="w-full border-b border-stone-200/80 bg-white/90 backdrop-blur-sm"
      aria-label="Dashboard navigation"
    >
      <div className="mx-auto max-w-7xl px-6">
        <div className="flex items-center justify-between gap-4 py-4">
          <div className="flex flex-col gap-0.5">
            <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-stone-400">
              v2.4.1 - monitoring
            </span>
            <h1 className="leading-none text-2xl font-black tracking-tight text-stone-900">
              Spam<span className="text-amber-600">Detector</span>
            </h1>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 rounded-full border border-stone-200 bg-stone-50 px-3 py-1.5">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-green-500" />
              </span>
            </div>
          </div>
        </div>

        <div
          className="flex items-center gap-0.5 overflow-x-auto"
          role="tablist"
          aria-label="Dashboard sections"
        >
          {TABS.map((tab) => {
            const isActive = activeTab === tab.id;

            return (
              <button
                key={tab.id}
                role="tab"
                aria-selected={isActive}
                onClick={() => startTransition(() => onTabChange(tab.id))}
                className={[
                  "group relative flex items-center gap-2 whitespace-nowrap px-4 py-3 text-sm font-semibold transition-colors duration-150 focus-visible:outline-none",
                  isActive
                    ? "text-amber-700"
                    : "text-stone-500 hover:text-stone-800",
                ].join(" ")}
                type="button"
              >
                <span className="text-[13px] opacity-70">{tab.icon}</span>
                {tab.label}
                {tab.count != null && (
                  <span
                    className={[
                      "rounded-full px-1.5 py-0.5 font-mono text-[10px] font-medium",
                      isActive
                        ? "bg-amber-100 text-amber-700"
                        : "bg-stone-100 text-stone-400",
                    ].join(" ")}
                  >
                    {tab.count}
                  </span>
                )}

                <span
                  className={[
                    "absolute bottom-0 left-0 h-0.5 w-full rounded-full bg-amber-600 transition-all duration-200",
                    isActive
                      ? "opacity-100"
                      : "opacity-0 group-hover:opacity-20",
                  ].join(" ")}
                />
              </button>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
