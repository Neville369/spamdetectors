import { FAMILY_COLORS, formatPercent, formatSeconds, titleize } from "../utils";

const METRIC_TONES = {
  amber: "border-amber-200 bg-amber-50 text-amber-900",
  teal: "border-teal-200 bg-teal-50 text-teal-900",
};

function PanelHeading({ subtitle, title }) {
  return (
    <div className="flex flex-col gap-2">
      <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-stone-400">
        Dashboard module
      </p>
      <div>
        <h2 className="text-xl font-black tracking-tight text-stone-900 sm:text-2xl">
          {title}
        </h2>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-stone-600">{subtitle}</p>
      </div>
    </div>
  );
}

function MetricCard({ label, tone, value }) {
  return (
    <div
      className={[
        "rounded-2xl border px-4 py-4 shadow-sm",
        METRIC_TONES[tone] ?? METRIC_TONES.teal,
      ].join(" ")}
    >
      <p className="text-xs font-medium uppercase tracking-[0.18em] text-stone-500">
        {label}
      </p>
      <p className="mt-2 text-xl font-black tracking-tight sm:text-2xl">{value}</p>
    </div>
  );
}

function ComparisonBars({ models }) {
  return (
    <div className="mt-6 space-y-4">
      {models.map((model, index) => (
        <div
          key={model.id}
          className="rounded-[24px] border border-stone-200 bg-stone-50/80 p-5 shadow-sm"
        >
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="max-w-2xl">
              <div className="flex flex-wrap items-center gap-3">
                <span className="rounded-full border border-stone-200 bg-white px-3 py-1 font-mono text-[11px] uppercase tracking-[0.18em] text-stone-500">
                  Rank {index + 1}
                </span>
                <span
                  className="inline-flex items-center gap-2 rounded-full border border-stone-200 bg-white px-3 py-1 text-sm font-semibold text-stone-700"
                >
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: FAMILY_COLORS[model.family] }}
                  />
                  {model.family}
                </span>
              </div>

              <h3 className="mt-4 text-xl font-black tracking-tight text-stone-900">
                {model.label}
              </h3>
              <p className="mt-2 text-sm leading-6 text-stone-600">{model.summary}</p>
              <p className="mt-3 text-sm font-medium text-stone-500">
                Feature space: {titleize(model.featureDescriptor)}
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3 lg:w-[340px] lg:grid-cols-1">
              <div className="rounded-2xl border border-stone-200 bg-white px-4 py-3">
                <p className="text-[11px] uppercase tracking-[0.18em] text-stone-400">
                  Weighted F1
                </p>
                <p className="mt-2 text-lg font-bold text-stone-900">
                  {formatPercent(model.f1)}
                </p>
              </div>
              <div className="rounded-2xl border border-stone-200 bg-white px-4 py-3">
                <p className="text-[11px] uppercase tracking-[0.18em] text-stone-400">
                  Accuracy
                </p>
                <p className="mt-2 text-lg font-bold text-stone-900">
                  {formatPercent(model.accuracy)}
                </p>
              </div>
              <div className="rounded-2xl border border-stone-200 bg-white px-4 py-3">
                <p className="text-[11px] uppercase tracking-[0.18em] text-stone-400">
                  Training
                </p>
                <p className="mt-2 text-lg font-bold text-stone-900">
                  {formatSeconds(model.trainingSeconds)}
                </p>
              </div>
            </div>
          </div>

          <div className="mt-5">
            <div className="flex items-center justify-between gap-3 text-sm">
              <span className="font-semibold text-stone-700">Weighted F1 score</span>
              <span className="font-mono text-stone-500">{formatPercent(model.f1)}</span>
            </div>
            <div className="mt-2 h-3 overflow-hidden rounded-full bg-stone-200">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  backgroundColor: FAMILY_COLORS[model.family],
                  width: `${Math.max(model.f1 * 100, 6)}%`,
                }}
              />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function Comparison({ loading = false, models = [] }) {
  const comparisonModels = [...models].sort((left, right) => right.f1 - left.f1);
  const bestClassical = comparisonModels.find((model) => model.family === "Classical ML");
  const bestDeepLearning = comparisonModels.find((model) => model.family === "Deep Learning");

  if (!comparisonModels.length) {
    return (
      <section className="mt-6">
        <article className="rounded-[28px] border border-stone-200/80 bg-white/90 p-6 shadow-[0_24px_70px_rgba(120,113,108,0.12)] sm:p-7">
          <PanelHeading
            title="Model Comparison"
            subtitle="Side-by-side ranking of classical ML against deep-learning baselines."
          />
          <p className="mt-6 text-sm leading-6 text-stone-500">
            {loading
              ? "Loading trained model metrics from the backend artifacts."
              : "No trained model metrics are available for the selected artifact profile yet."}
          </p>
        </article>
      </section>
    );
  }

  return (
    <section className="mt-6">
      <article className="rounded-[28px] border border-stone-200/80 bg-white/90 p-6 shadow-[0_24px_70px_rgba(120,113,108,0.12)] sm:p-7">
        <PanelHeading
          title="Model Comparison"
          subtitle="Side-by-side ranking of classical ML against deep-learning baselines."
        />

        <div className="mt-6 grid gap-4 md:grid-cols-2">
          {bestClassical ? (
            <MetricCard
              label="Best classical"
              value={`${titleize(bestClassical.modelType)} (${formatPercent(bestClassical.f1)})`}
              tone="teal"
            />
          ) : null}
          {bestDeepLearning ? (
            <MetricCard
              label="Best deep learning"
              value={`${titleize(bestDeepLearning.modelType)} (${formatPercent(bestDeepLearning.f1)})`}
              tone="amber"
            />
          ) : null}
        </div>

        <ComparisonBars models={comparisonModels} />
      </article>
    </section>
  );
}
