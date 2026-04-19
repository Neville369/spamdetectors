const rawBaseUrl = import.meta.env.VITE_BACKEND_BASE_URL?.trim() ?? "";
const resolvedBaseUrl = rawBaseUrl.replace(/\/+$/, "");
const artifactProfile = import.meta.env.VITE_BACKEND_ARTIFACT_PROFILE ?? "sample";

const mlArtifactDirs = {
  main: import.meta.env.VITE_BACKEND_MAIN_ARTIFACT_DIR ?? "../backend/artifacts",
  sample: import.meta.env.VITE_BACKEND_SAMPLE_ARTIFACT_DIR ?? "../backend/artifacts_small",
};

const dlArtifactDirs = {
  main: import.meta.env.VITE_BACKEND_DL_MAIN_ARTIFACT_DIR ?? "../backend/dl_artifacts",
  sample:
    import.meta.env.VITE_BACKEND_DL_SAMPLE_ARTIFACT_DIR ?? "../backend/dl_artifacts_small",
};

export const backendConfig = {
  artifactProfile,
  baseUrl: resolvedBaseUrl,
  baseUrlLabel: resolvedBaseUrl || "same-origin (/api)",
  dlArtifactDirs,
  mlArtifactDirs,
  selectedDlArtifacts: dlArtifactDirs[artifactProfile] || dlArtifactDirs.sample,
  selectedMlArtifacts: mlArtifactDirs[artifactProfile] || mlArtifactDirs.sample,
};

export function buildBackendUrl(pathname = "") {
  const normalizedPath = String(pathname).replace(/^\/+/, "");

  if (!normalizedPath) {
    return backendConfig.baseUrl || "/";
  }

  return backendConfig.baseUrl
    ? `${backendConfig.baseUrl}/${normalizedPath}`
    : `/${normalizedPath}`;
}
