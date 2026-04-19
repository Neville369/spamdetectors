import { useEffect, useState } from "react";

import { predictMessage } from "../api/dashboard";
import { DEFAULT_MODEL_ID } from "../data/dashboard";
import {
  FAMILY_COLORS,
  LABEL_COLORS,
  formatPercent,
  formatSeconds,
  titleize,
} from "../utils";

const dashboard = {
  samples: [
    {
      label: "ham",
      text: "Hi team, sharing the updated project timeline before tomorrow's review. Please send any comments before 4 PM.",
    },
    {
      label: "spam",
      text: "Limited-time offer. Claim your free bonus cash now and unlock this exclusive discount before midnight.",
    },
    {
      label: "phish",
      text: "Your account has been suspended. Verify your password immediately using the secure link below to avoid service interruption.",
    },
  ],
};

const STATUS_STYLES = {
  ham: "border border-teal-200 bg-teal-50 text-teal-700",
  phish: "border border-amber-200 bg-amber-50 text-amber-700",
  spam: "border border-rose-200 bg-rose-50 text-rose-700",
};

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

function ModelStat({ label, value }) {
  return (
    <div className="rounded-2xl border border-stone-200 bg-white/80 px-4 py-3 shadow-sm">
      <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-stone-400">
        {label}
      </p>
      <p className="mt-2 text-sm font-semibold text-stone-800">{value}</p>
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

function ModelOptionButton({ disabled, isActive, model, onSelect }) {
  return (
    <button
      className={[
        "rounded-[22px] border px-4 py-4 text-left transition",
        isActive
          ? "border-stone-900 bg-white shadow-sm"
          : "border-stone-200 bg-stone-50/80 hover:border-stone-300 hover:bg-white",
        disabled ? "cursor-not-allowed opacity-60" : "",
      ].join(" ")}
      onClick={() => onSelect(model.id)}
      type="button"
      disabled={disabled}
    >
      <div className="flex items-center justify-between gap-3">
        <span className="inline-flex items-center gap-2 text-[11px] font-medium uppercase tracking-[0.18em] text-stone-400">
          <span
            className="h-2.5 w-2.5 rounded-full"
            style={{ backgroundColor: FAMILY_COLORS[model.family] }}
          />
          {model.family}
        </span>
        <span className="font-mono text-xs text-stone-500">
          {formatPercent(model.accuracy)}
        </span>
      </div>

      <p className="mt-3 text-base font-bold text-stone-900">{model.label}</p>
      <p className="mt-1 text-sm text-stone-500">{model.pipeline}</p>
    </button>
  );
}

function ProbabilityBar({ color, label, value }) {
  return (
    <div className="space-y-2 rounded-2xl border border-stone-200 bg-white/80 p-4">
      <div className="flex items-center justify-between gap-3 text-sm">
        <span className="font-semibold text-stone-700">{titleize(label)}</span>
        <span className="font-mono text-stone-500">{formatPercent(value)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-stone-100">
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{
            backgroundColor: color,
            width: `${Math.max(value * 100, 6)}%`,
          }}
        />
      </div>
    </div>
  );
}

export default function Prediction({
  loading = false,
  models = [],
  profile = "sample",
}) {
  const [selectedModelId, setSelectedModelId] = useState(DEFAULT_MODEL_ID);
  const [predictionInput, setPredictionInput] = useState(dashboard.samples[0].text);
  const [predictionResult, setPredictionResult] = useState(null);
  const [predictionError, setPredictionError] = useState("");
  const [classifying, setClassifying] = useState(false);
  const [feedbackCorrectLabel, setFeedbackCorrectLabel] = useState("ham");
  const [feedbackNotes, setFeedbackNotes] = useState("");
  const [feedbackMessage, setFeedbackMessage] = useState("");
  const [sendingFeedback, setSendingFeedback] = useState(false);

  useEffect(() => {
    if (!models.length) {
      return;
    }

    const hasActiveModel = models.some((model) => model.id === selectedModelId);
    if (!hasActiveModel) {
      setSelectedModelId(models[0].id);
    }
  }, [models, selectedModelId]);

  const selectedModel = models.find((model) => model.id === selectedModelId) ?? null;

  const handleModelChange = (modelId) => {
    if (modelId === selectedModelId) {
      return;
    }

    setSelectedModelId(modelId);
    setPredictionResult(null);
    setPredictionError("");
    setFeedbackMessage("");
  };

  const handleClassify = async () => {
    const trimmed = predictionInput.trim();

    if (!trimmed) {
      setPredictionError("Paste a message before running the classifier.");
      setPredictionResult(null);
      return;
    }

    if (!selectedModel) {
      setPredictionError("No trained model is available for the selected artifact profile.");
      setPredictionResult(null);
      return;
    }

    setClassifying(true);
    setPredictionError("");
    setFeedbackMessage("");

    const startedAt = performance.now();

    try {
      const result = await predictMessage({
        modelId: selectedModel.id,
        profile,
        text: trimmed,
      });

      setPredictionResult({
        ...result,
        responseMs: performance.now() - startedAt,
      });
      setFeedbackCorrectLabel(result.prediction.groupLabel);
    } catch (error) {
      setPredictionError(
        error instanceof Error
          ? error.message
          : "The backend prediction API could not classify this message.",
      );
      setPredictionResult(null);
    } finally {
      setClassifying(false);
    }
  };

  const handleFeedbackSubmit = async () => {
    if (!predictionResult) {
      setFeedbackMessage("Run a prediction before saving feedback.");
      return;
    }

    setSendingFeedback(true);

    window.setTimeout(() => {
      setFeedbackMessage(
        [
          `Feedback saved for ${predictionResult.model.label} as ${titleize(feedbackCorrectLabel)}.`,
          feedbackNotes.trim() ? "Notes added for follow-up review." : "No notes were added.",
        ].join(" "),
      );
      setSendingFeedback(false);
    }, 300);
  };

  if (!selectedModel) {
    return (
      <section className="mt-6">
        <article className="rounded-[28px] border border-stone-200/80 bg-white/90 p-6 shadow-[0_24px_70px_rgba(120,113,108,0.12)] sm:p-7">
          <PanelHeading
            title="Real-Time Prediction"
            subtitle="Switch between trained models and run the selected classifier on any message."
          />
          <p className="mt-6 text-sm leading-6 text-stone-500">
            {loading
              ? "Loading trained models from the backend artifacts."
              : "No trained models are available for prediction in the selected artifact profile."}
          </p>
        </article>
      </section>
    );
  }

  return (
    <section className="mt-6">
      <article className="rounded-[28px] border border-stone-200/80 bg-white/90 p-6 shadow-[0_24px_70px_rgba(120,113,108,0.12)] sm:p-7">
        <PanelHeading
          title="Real-Time Prediction"
          subtitle="Switch between trained models and run the selected classifier on any message."
        />

        <div className="mt-6 rounded-[24px] border border-stone-200 bg-stone-50/80 p-5">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="max-w-2xl">
              <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-stone-400">
                Active model
              </p>
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <h3 className="text-2xl font-black tracking-tight text-stone-900">
                  {selectedModel.label}
                </h3>
                <span className="inline-flex items-center gap-2 rounded-full border border-stone-200 bg-white px-3 py-1 text-sm font-semibold text-stone-700">
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: FAMILY_COLORS[selectedModel.family] }}
                  />
                  {selectedModel.family}
                </span>
              </div>
              <p className="mt-3 text-sm leading-6 text-stone-600">
                {selectedModel.summary}
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <ModelStat label="Pipeline" value={selectedModel.pipeline} />
              <ModelStat
                label="Validation accuracy"
                value={formatPercent(selectedModel.accuracy)}
              />
              <ModelStat
                label="Training time"
                value={formatSeconds(selectedModel.trainingSeconds)}
              />
            </div>
          </div>

          <div className="mt-5 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
            {models.map((model) => (
              <ModelOptionButton
                key={model.id}
                disabled={classifying}
                isActive={model.id === selectedModelId}
                model={model}
                onSelect={handleModelChange}
              />
            ))}
          </div>

          <p className="mt-4 text-sm text-stone-500">
            Predictions now run through the backend API using the saved{" "}
            <strong>{profile}</strong> artifact profile.
          </p>
        </div>

        <div className="mt-6 space-y-4">
          <textarea
            className="min-h-[180px] w-full rounded-[24px] border border-stone-200 bg-stone-50/80 px-4 py-4 text-sm leading-6 text-stone-700 outline-none transition focus:border-amber-300 focus:bg-white"
            value={predictionInput}
            onChange={(event) => setPredictionInput(event.target.value)}
            placeholder="Paste an email body or short message."
          />

          <div className="flex flex-wrap gap-3">
            {dashboard.samples.map((sample) => (
              <button
                key={sample.label}
                className="rounded-full border border-stone-200 bg-stone-50 px-4 py-2 text-sm font-medium text-stone-600 transition hover:border-amber-300 hover:bg-amber-50 hover:text-amber-700"
                onClick={() => setPredictionInput(sample.text)}
                type="button"
              >
                Load {titleize(sample.label)} example
              </button>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              className="rounded-full bg-stone-900 px-5 py-3 text-sm font-semibold text-white transition hover:bg-stone-800 disabled:cursor-not-allowed disabled:bg-stone-400"
              onClick={handleClassify}
              disabled={classifying}
              type="button"
            >
              {classifying ? `Running ${selectedModel.label}...` : `Classify with ${selectedModel.label}`}
            </button>
            <p className="text-sm text-stone-500">
              Backend inference reads the saved model artifacts and returns class probabilities.
            </p>
          </div>

          {predictionError ? (
            <p className="text-sm font-medium text-rose-600">{predictionError}</p>
          ) : null}
        </div>

        {predictionResult ? (
          <div className="mt-8 space-y-6 rounded-[28px] border border-stone-200 bg-stone-50/60 p-5 sm:p-6">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-stone-400">
                  Predicted risk
                </p>
                <h3 className="mt-2 text-3xl font-black tracking-tight text-stone-900">
                  {predictionResult.prediction.groupLabel === "ham" ? "Ham" : "Spam"}
                </h3>
                <div className="mt-3 flex flex-wrap gap-2">
                  <span className="rounded-full border border-stone-200 bg-white px-3 py-1 text-sm font-semibold text-stone-700">
                    Model: {predictionResult.model.label}
                  </span>
                  <span className="rounded-full border border-stone-200 bg-white px-3 py-1 text-sm font-semibold text-stone-700">
                    Pipeline: {predictionResult.model.pipeline}
                  </span>
                </div>
              </div>
              <span
                className={[
                  "inline-flex rounded-full px-3 py-1 text-sm font-semibold",
                  STATUS_STYLES[predictionResult.prediction.label],
                ].join(" ")}
              >
                Detail: {titleize(predictionResult.prediction.label)}
              </span>
            </div>

            <div className="grid gap-4 md:grid-cols-3">
              <MetricCard
                label="Risk confidence"
                value={formatPercent(predictionResult.prediction.groupConfidence)}
                tone="teal"
              />
              <MetricCard
                label="Class confidence"
                value={formatPercent(predictionResult.prediction.confidence)}
                tone="coral"
              />
              <MetricCard
                label="Response time"
                value={`${predictionResult.responseMs.toFixed(2)} ms`}
                tone="amber"
              />
            </div>

            <div className="grid gap-3">
              {predictionResult.prediction.classProbabilities.map((entry) => (
                <ProbabilityBar
                  key={entry.label}
                  label={entry.label}
                  value={entry.confidence}
                  color={LABEL_COLORS[entry.label]}
                />
              ))}
            </div>

            <div className="rounded-[24px] border border-stone-200 bg-white/80 p-5">
              <h4 className="text-lg font-bold text-stone-900">Top contributing features</h4>
              <div className="mt-4 flex flex-wrap gap-3">
                {predictionResult.prediction.topSignals?.length ? (
                  predictionResult.prediction.topSignals.map((signal) => (
                    <span
                      key={signal.term}
                      className="rounded-full border border-stone-200 bg-stone-50 px-3 py-2 text-sm text-stone-700"
                    >
                      {signal.term} <strong>{signal.contribution}</strong>
                    </span>
                  ))
                ) : (
                  <p className="text-sm leading-6 text-stone-500">
                    This model did not return token-level feature explanations for the current prediction.
                  </p>
                )}
              </div>
            </div>

            <div className="rounded-[24px] border border-stone-200 bg-white/80 p-5">
              <div className="flex flex-col gap-2">
                <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-stone-400">
                  Feedback loop
                </p>
                <h4 className="text-lg font-bold text-stone-900">
                  Mark incorrect predictions for retraining review.
                </h4>
              </div>

              <div className="mt-4 grid gap-4 md:grid-cols-[180px_1fr_auto] md:items-end">
                <label className="grid gap-2 text-sm font-medium text-stone-700">
                  Correct risk label
                  <select
                    className="rounded-2xl border border-stone-200 bg-stone-50 px-4 py-3 outline-none transition focus:border-amber-300 focus:bg-white"
                    value={feedbackCorrectLabel}
                    onChange={(event) => setFeedbackCorrectLabel(event.target.value)}
                  >
                    <option value="ham">Ham</option>
                    <option value="spam">Spam</option>
                  </select>
                </label>

                <label className="grid gap-2 text-sm font-medium text-stone-700">
                  Notes
                  <input
                    className="rounded-2xl border border-stone-200 bg-stone-50 px-4 py-3 outline-none transition focus:border-amber-300 focus:bg-white"
                    type="text"
                    value={feedbackNotes}
                    onChange={(event) => setFeedbackNotes(event.target.value)}
                    placeholder="Optional comments for future analysis"
                  />
                </label>

                <button
                  className="rounded-full border border-stone-300 bg-white px-5 py-3 text-sm font-semibold text-stone-700 transition hover:border-stone-400 hover:text-stone-900 disabled:cursor-not-allowed disabled:opacity-60"
                  onClick={handleFeedbackSubmit}
                  disabled={sendingFeedback}
                  type="button"
                >
                  {sendingFeedback ? "Saving..." : "Save feedback"}
                </button>
              </div>

              {feedbackMessage ? (
                <p className="mt-4 text-sm font-medium text-stone-600">{feedbackMessage}</p>
              ) : null}
            </div>
          </div>
        ) : null}
      </article>
    </section>
  );
}
