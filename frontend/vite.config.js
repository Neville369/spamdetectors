import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

function resolveProxyTarget(env) {
  const candidates = [env.VITE_BACKEND_PROXY_TARGET, env.VITE_BACKEND_BASE_URL];
  const explicitTarget = candidates.find((value) => value?.trim().startsWith("http"));

  return explicitTarget?.trim() || "http://127.0.0.1:8000";
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  return {
    plugins: [react(), tailwindcss()],
    server: {
      proxy: {
        "/api": {
          changeOrigin: true,
          target: resolveProxyTarget(env),
        },
      },
    },
  };
});
