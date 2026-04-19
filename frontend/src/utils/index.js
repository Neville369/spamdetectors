export const LABEL_COLORS = {
  ham: "#0f766e",
  phish: "#d97706",
  spam: "#dc2626",
};

export const FAMILY_COLORS = {
  "Classical ML": "#0f766e",
  "Deep Learning": "#b45309",
};

export function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

export function formatSeconds(value) {
  return `${Number(value || 0).toFixed(3)} s`;
}

export function titleize(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}
