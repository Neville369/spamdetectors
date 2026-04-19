import { LABEL_COLORS, titleize } from "../utils";

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

function DonutCard({ segments, title }) {
  const total = segments.reduce((sum, segment) => sum + segment.count, 0);
  let startAngle = 0;

  const circleSegments = segments.map((segment) => {
    const angle = total ? (segment.count / total) * 360 : 0;
    const currentStart = startAngle;
    startAngle += angle;

    return {
      ...segment,
      angle,
      startAngle: currentStart,
    };
  });

  return (
    <div className="rounded-[28px] border border-stone-200 bg-white/80 p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-lg font-bold text-stone-900">{title}</h3>
          <p className="mt-1 text-sm text-stone-500">{total.toLocaleString()} messages</p>
        </div>
        <div className="relative h-32 w-32 shrink-0">
          <div
            className="h-full w-full rounded-full"
            style={{
              background: `conic-gradient(${circleSegments
                .map(
                  (segment) =>
                    `${segment.color} ${segment.startAngle}deg ${segment.startAngle + segment.angle}deg`,
                )
                .join(", ")})`,
            }}
          />
          <div className="absolute inset-[22%] grid place-items-center rounded-full bg-white shadow-inner">
            <span className="text-center text-xs font-semibold uppercase tracking-[0.18em] text-stone-500">
              Mix
            </span>
          </div>
        </div>
      </div>

      <div className="mt-5 space-y-3">
        {segments.map((segment) => (
          <div key={segment.label} className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <span
                className="h-3 w-3 rounded-full"
                style={{ backgroundColor: segment.color }}
              />
              <span className="text-sm font-medium text-stone-700">
                {titleize(segment.label)}
              </span>
            </div>
            <div className="text-right">
              <p className="text-sm font-semibold text-stone-900">
                {segment.count.toLocaleString()}
              </p>
              <p className="text-xs text-stone-500">
                {total ? ((segment.count / total) * 100).toFixed(1) : "0.0"}%
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Analytics({ analytics, loading = false }) {
  if (!analytics) {
    return (
      <section className="mt-6 rounded-[28px] border border-stone-200/80 bg-white/90 p-6 shadow-[0_24px_70px_rgba(120,113,108,0.12)] sm:p-7">
        <PanelHeading
          title="Data Insights"
          subtitle="Quick distribution snapshots for class balance and operational grouping."
        />
        <p className="mt-6 text-sm leading-6 text-stone-500">
          {loading
            ? "Loading analytics from the backend artifacts."
            : "No analytics data is available for the selected artifact profile."}
        </p>
      </section>
    );
  }

  return (
    <section className="mt-6 rounded-[28px] border border-stone-200/80 bg-white/90 p-6 shadow-[0_24px_70px_rgba(120,113,108,0.12)] sm:p-7">
      <PanelHeading
        title="Data Insights"
        subtitle="Quick distribution snapshots for class balance and operational grouping."
      />

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <DonutCard
          title="Class balance"
          segments={Object.entries(analytics.classCounts).map(([label, count]) => ({
            color: LABEL_COLORS[label],
            count,
            label,
          }))}
        />
        <DonutCard
          title="Operational grouping"
          segments={Object.entries(analytics.groupedClassCounts).map(
            ([label, count]) => ({
              color: label === "ham" ? "#0f766e" : "#dc2626",
              count,
              label,
            }),
          )}
        />
      </div>
    </section>
  );
}
