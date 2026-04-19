import { Fragment, useEffect, useState, useTransition } from "react";

import { DEFAULT_MODEL_ID, PERFORMANCE_LABELS } from "../data/dashboard";
import {
  FAMILY_COLORS,
  LABEL_COLORS,
  formatPercent,
  formatSeconds,
  titleize,
} from "../utils";

const METRIC_TONES = {
  amber: "border-amber-200 bg-amber-50 text-amber-900",
  coral: "border-rose-200 bg-rose-50 text-rose-900",
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
      <p className="mt-2 text-2xl font-black tracking-tight">{value}</p>
    </div>
  );
}

function Badge({ children, color = null }) {
  return (
    <span
      className="inline-flex items-center gap-2 rounded-full border border-stone-200 bg-white px-3 py-1.5 text-sm font-semibold text-stone-700 shadow-sm"
      style={color ? { borderColor: `${color}33` } : undefined}
    >
      {color ? (
        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
      ) : null}
      {children}
    </span>
  );
}

function hexToRgb(hex) {
  const normalized = hex.replace("#", "");
  const expanded =
    normalized.length === 3
      ? normalized
          .split("")
          .map((char) => `${char}${char}`)
          .join("")
      : normalized;
  const value = Number.parseInt(expanded, 16);

  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function alphaColor(hex, alpha) {
  const [red, green, blue] = hexToRgb(hex);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function ConfusionMatrix({ labels, matrix }) {
  const maxValue = Math.max(...matrix.flat(), 1);

  return (
    <div className="mt-6 rounded-[24px] border border-stone-200 bg-stone-50/80 p-5 shadow-sm">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <Badge>Rows = actual labels</Badge>
        <Badge>Columns = predicted labels</Badge>
      </div>

      <div className="overflow-x-auto">
        <div className="grid min-w-[340px] grid-cols-[96px_repeat(3,minmax(72px,1fr))] gap-3">
          <div />
          {labels.map((label) => (
            <div
              key={`header-${label}`}
              className="rounded-2xl border border-stone-200 bg-white px-3 py-3 text-center shadow-sm"
            >
              <p className="text-[11px] uppercase tracking-[0.18em] text-stone-400">
                Predicted
              </p>
              <p className="mt-1 font-semibold text-stone-800">{titleize(label)}</p>
            </div>
          ))}

          {labels.map((rowLabel, rowIndex) => (
            <Fragment key={rowLabel}>
              <div className="flex items-center rounded-2xl border border-stone-200 bg-white px-3 py-3 shadow-sm">
                <div>
                  <p className="text-[11px] uppercase tracking-[0.18em] text-stone-400">
                    Actual
                  </p>
                  <p className="mt-1 font-semibold text-stone-800">{titleize(rowLabel)}</p>
                </div>
              </div>

              {labels.map((columnLabel, columnIndex) => {
                const value = matrix[rowIndex][columnIndex];
                const intensity = value / maxValue;
                const isDiagonal = rowIndex === columnIndex;

                return (
                  <div
                    key={`${rowLabel}-${columnLabel}`}
                    className="rounded-2xl border px-3 py-4 text-center shadow-sm transition-transform duration-150 hover:-translate-y-0.5"
                    style={{
                      backgroundColor: isDiagonal
                        ? alphaColor(LABEL_COLORS[rowLabel], 0.16 + intensity * 0.44)
                        : alphaColor("#78716c", 0.08 + intensity * 0.12),
                      borderColor: isDiagonal ? alphaColor(LABEL_COLORS[rowLabel], 0.42) : undefined,
                    }}
                  >
                    <p className="text-2xl font-black tracking-tight text-stone-900">{value}</p>
                    <p className="mt-1 text-[11px] uppercase tracking-[0.18em] text-stone-500">
                      {isDiagonal ? "Correct" : "Off-diagonal"}
                    </p>
                  </div>
                );
              })}
            </Fragment>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function Performance({
  labels = PERFORMANCE_LABELS,
  loading = false,
  models = [],
}) {
  const [selectedModelId, setSelectedModelId] = useState(DEFAULT_MODEL_ID);
  const [, startTransition] = useTransition();

  useEffect(() => {
    if (!models.length) {
      return;
    }

    const hasActiveModel = models.some((model) => model.id === selectedModelId);
    if (!hasActiveModel) {
      setSelectedModelId(models[0].id);
    }
  }, [models, selectedModelId]);

  if (!models.length) {
    return (
      <section className="mt-6 rounded-[28px] border border-stone-200/80 bg-white/90 p-6 shadow-[0_24px_70px_rgba(120,113,108,0.12)] sm:p-7">
        <PanelHeading
          title="Model Performance"
          subtitle="Switch across all classical and deep-learning runs using the same dashboard frame."
        />
        <p className="mt-6 text-sm leading-6 text-stone-500">
          {loading
            ? "Loading performance metrics from the backend artifacts."
            : "No trained model performance data is available for the selected artifact profile."}
        </p>
      </section>
    );
  }

  const selectedModel =
    models.find((model) => model.id === selectedModelId) ?? models[0];

  return (
    <section className="mt-6 grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
      <article className="rounded-[28px] border border-stone-200/80 bg-white/90 p-6 shadow-[0_24px_70px_rgba(120,113,108,0.12)] sm:p-7">
        <PanelHeading
          title="Model Performance"
          subtitle="Switch across all classical and deep-learning runs using the same dashboard frame."
        />

        <label className="mt-6 grid gap-2 text-sm font-medium text-stone-700">
          <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-stone-400">
            Selected model
          </span>
          <select
            className="rounded-[20px] border border-stone-200 bg-stone-50/80 px-4 py-3 text-sm text-stone-700 outline-none transition focus:border-amber-300 focus:bg-white"
            value={selectedModel.id}
            onChange={(event) => {
              startTransition(() => setSelectedModelId(event.target.value));
            }}
          >
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.family} | {model.label} | {model.featureDescriptor}
              </option>
            ))}
          </select>
        </label>

        <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard
            label="Accuracy"
            value={formatPercent(selectedModel.accuracy)}
            tone="teal"
          />
          <MetricCard
            label="Precision"
            value={formatPercent(selectedModel.precision)}
            tone="amber"
          />
          <MetricCard
            label="Recall"
            value={formatPercent(selectedModel.recall)}
            tone="coral"
          />
          <MetricCard label="F1 score" value={formatPercent(selectedModel.f1)} tone="teal" />
        </div>

        <div className="mt-6 flex flex-wrap gap-3">
          <Badge color={FAMILY_COLORS[selectedModel.family]}>{selectedModel.family}</Badge>
          <Badge>Training time: {formatSeconds(selectedModel.trainingSeconds)}</Badge>
          <Badge>
            Train/Test: {selectedModel.trainRows.toLocaleString()}/
            {selectedModel.testRows.toLocaleString()}
          </Badge>
          {selectedModel.featureCount ? (
            <Badge>Feature count: {selectedModel.featureCount.toLocaleString()}</Badge>
          ) : null}
          {selectedModel.vocabSize ? (
            <Badge>Vocab size: {selectedModel.vocabSize.toLocaleString()}</Badge>
          ) : null}
        </div>

        <p className="mt-5 text-sm leading-6 text-stone-600">{selectedModel.summary}</p>

        <div className="mt-6 grid gap-4 md:grid-cols-3">
          {labels.map((label) => {
            const classReport = selectedModel.classificationReport[label];

            return (
              <div
                className="rounded-[24px] border border-stone-200 bg-stone-50/80 p-5 shadow-sm"
                key={label}
              >
                <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-stone-400">
                  {titleize(label)}
                </p>
                <h3 className="mt-3 text-3xl font-black tracking-tight text-stone-900">
                  {formatPercent(classReport["f1-score"])}
                </h3>
                <p className="mt-2 text-sm leading-6 text-stone-500">
                  Precision {formatPercent(classReport.precision)} | Recall{" "}
                  {formatPercent(classReport.recall)}
                </p>
              </div>
            );
          })}
        </div>
      </article>

      <article className="rounded-[28px] border border-stone-200/80 bg-white/90 p-6 shadow-[0_24px_70px_rgba(120,113,108,0.12)] sm:p-7">
        <PanelHeading
          title="Confusion Matrix"
          subtitle="Heatmap view of predicted vs actual classes for the selected run."
        />

        <div className="mt-6 flex flex-wrap gap-3">
          <Badge color={FAMILY_COLORS[selectedModel.family]}>{selectedModel.label}</Badge>
          <Badge>Validation set: {selectedModel.testRows.toLocaleString()} messages</Badge>
        </div>

        <ConfusionMatrix labels={labels} matrix={selectedModel.confusionMatrix} />
      </article>
    </section>
  );
}
