#!/usr/bin/env bash
set -euo pipefail

PORT="${VITE_PORT:-5174}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared not found. Install it first."
  exit 1
fi

if ! command -v npx >/dev/null 2>&1; then
  echo "npx not found. Install Node.js first."
  exit 1
fi

PIDS="$(lsof -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true)"
if [[ -n "${PIDS}" ]]; then
  echo "Killing processes on port ${PORT}: ${PIDS}"
  if ! kill -9 ${PIDS} 2>/dev/null; then
    echo "Failed to kill existing process on port ${PORT}. Stop it manually and re-run."
    exit 1
  fi
fi

LOG_VITE="/tmp/vite-dev-${PORT}.log"
LOG_CF="/tmp/cf-tunnel-${PORT}.log"

nohup npx vite --port "${PORT}" --strictPort > "${LOG_VITE}" 2>&1 &
VITE_PID=$!

for _ in {1..30}; do
  if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

nohup cloudflared tunnel --url "http://localhost:${PORT}" --no-autoupdate > "${LOG_CF}" 2>&1 &
CF_PID=$!

for _ in {1..60}; do
  URL="$(grep -o "https://[a-z0-9-]*\\.trycloudflare.com" "${LOG_CF}" | grep -v "api.trycloudflare.com" | tail -n 1 || true)"
  if [[ -n "${URL}" ]]; then
    echo ""
    echo "Tunnel URL: ${URL}"
    echo "Vite log: ${LOG_VITE}"
    echo "Tunnel log: ${LOG_CF}"
    exit 0
  fi
  sleep 0.5
done

echo "Tunnel URL not found yet. Check logs:"
echo "${LOG_CF}"
exit 1
