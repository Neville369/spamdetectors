import { backendConfig, buildBackendUrl } from "../config/backend";

function resolveBrowserOrigin() {
  return typeof window !== "undefined" ? window.location.origin : "http://localhost";
}

function resolveBackendUrl(pathname = "") {
  return new URL(buildBackendUrl(pathname), resolveBrowserOrigin());
}

function extractErrorMessage(payload, fallbackMessage) {
  if (typeof payload === "string") {
    return payload || fallbackMessage;
  }

  return payload?.detail || payload?.message || payload?.error || fallbackMessage;
}

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const fallbackMessage =
      response.status === 404
        ? `${response.url} returned 404. The frontend may be pointed at the wrong backend service.`
        : `Request to ${response.url} failed with ${response.status} ${response.statusText}.`;
    const message = extractErrorMessage(payload, fallbackMessage);
    throw new Error(message);
  }

  return payload;
}

async function requestBackend(input, init) {
  const url = input instanceof URL ? input : resolveBackendUrl(String(input));

  try {
    const response = await fetch(url, init);
    return await parseResponse(response);
  } catch (error) {
    if (error instanceof Error && error.name === "TypeError") {
      throw new Error(`Could not reach backend at ${url.toString()}. ${error.message}`);
    }

    throw error;
  }
}

export async function fetchDashboard(profile = backendConfig.artifactProfile) {
  const url = resolveBackendUrl("api/dashboard");
  url.searchParams.set("profile", profile);

  return requestBackend(url, {
    headers: {
      Accept: "application/json",
    },
  });
}

export async function predictMessage({
  modelId,
  profile = backendConfig.artifactProfile,
  text,
}) {
  return requestBackend(resolveBackendUrl("api/predict"), {
    body: JSON.stringify({
      modelId,
      profile,
      text,
    }),
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    method: "POST",
  });
}
