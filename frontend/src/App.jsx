import { useEffect, useState } from "react";

import { fetchDashboard } from "./api/dashboard";
import Analytics from "./component/Analytics";
import Comparison from "./component/Comparison";
import NavBar from "./component/NavBar";
import Performance from "./component/Performance";
import Prediction from "./component/Prediction";
import { FALLBACK_DASHBOARD } from "./data/dashboard";
import { backendConfig } from "./config/backend";

const TAB_DETAILS = {
  analytics: {
    copy: "Dataset-level analytics can live here once you are ready to chart message volume, label drift, and feature activity.",
    title: "Analytics",
  },
  comparison: {
    copy: "Use this area for side-by-side model benchmarking once the comparison dashboard is ready.",
    title: "Comparison",
  },
  performance: {
    copy: "Reserve this section for validation metrics, confusion matrices, and latency breakdowns across the trained models.",
    title: "Performance",
  },
};

function PlaceholderPanel({ activeTab }) {
  const details = TAB_DETAILS[activeTab];

  if (!details) {
    return null;
  }

  return (
    <section className="mt-6 rounded-[28px] border border-stone-200/80 bg-white/90 p-8 shadow-[0_24px_70px_rgba(120,113,108,0.12)]">
      <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-stone-400">
        Coming next
      </p>
      <h2 className="mt-3 text-3xl font-black tracking-tight text-stone-900">
        {details.title}
      </h2>
      <p className="mt-3 max-w-2xl text-sm leading-7 text-stone-600">{details.copy}</p>
    </section>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState("prediction");
  const [dashboardData, setDashboardData] = useState(FALLBACK_DASHBOARD);
  const [dashboardLoading, setDashboardLoading] = useState(true);
  const [dashboardError, setDashboardError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadDashboard() {
      setDashboardLoading(true);
      setDashboardError("");

      try {
        const payload = await fetchDashboard(backendConfig.artifactProfile);
        if (!cancelled) {
          setDashboardData(payload);
        }
      } catch (error) {
        if (!cancelled) {
          setDashboardError(
            error instanceof Error
              ? error.message
              : "The backend dashboard API could not be reached.",
          );
        }
      } finally {
        if (!cancelled) {
          setDashboardLoading(false);
        }
      }
    }

    loadDashboard();

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(251,191,36,0.18),_transparent_28%),linear-gradient(180deg,_#f8f5ef_0%,_#fcfaf6_48%,_#efe4d2_100%)] px-4 py-6 sm:px-6 lg:px-10 lg:py-12">
      <div className="mx-auto max-w-7xl">
        <NavBar activeTab={activeTab} onTabChange={setActiveTab} />
        {dashboardLoading ? (
          <div className="mt-4 rounded-2xl border border-stone-200/80 bg-white/80 px-4 py-3 text-sm text-stone-500 shadow-sm">
            Loading backend artifacts for the <strong>{backendConfig.artifactProfile}</strong>{" "}
            profile.
          </div>
        ) : null}
        {dashboardError ? (
          <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50/90 px-4 py-3 text-sm text-amber-800 shadow-sm">
            Backend API unavailable. Showing fallback dashboard data instead. Details:{" "}
            {dashboardError} Target: <strong>{backendConfig.baseUrlLabel}</strong>.
          </div>
        ) : null}
        {!dashboardError && dashboardData.warnings?.length ? (
          <div className="mt-4 rounded-2xl border border-stone-200/80 bg-white/80 px-4 py-3 text-sm text-stone-600 shadow-sm">
            Backend notes: {dashboardData.warnings[0]}
          </div>
        ) : null}
        {activeTab === "prediction" ? (
          <Prediction
            labels={dashboardData.metrics?.labels}
            loading={dashboardLoading}
            models={dashboardData.metrics?.models}
            profile={dashboardData.profile ?? backendConfig.artifactProfile}
          />
        ) : activeTab === "performance" ? (
          <Performance
            labels={dashboardData.metrics?.labels}
            loading={dashboardLoading}
            models={dashboardData.metrics?.models}
          />
        ) : activeTab === "analytics" ? (
          <Analytics
            analytics={dashboardData.analytics}
            loading={dashboardLoading}
          />
        ) : activeTab === "comparison" ? (
          <Comparison
            loading={dashboardLoading}
            models={dashboardData.metrics?.models}
          />
        ) : (
          <PlaceholderPanel activeTab={activeTab} />
        )}
      </div>
    </main>
  );
}

export default App;
