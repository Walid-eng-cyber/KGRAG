/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI service (uvicorn server.main:app --port 8000) owns /api.
const API = process.env.KGRAG_API ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": API } },
  test: { environment: "node" },
});
