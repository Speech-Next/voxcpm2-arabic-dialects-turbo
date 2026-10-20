#!/bin/bash
# The image's command: `http` (the OpenAI-compatible server, the default) or `ws` (the WebSocket server), then that
# server's options; it listens on 0.0.0.0:$PORT unless --host / --port are given. Anything else runs as given
# (bash, python -m voxcpm_kernels.cuda_ext, ...).
set -euo pipefail
case "${1:-http}" in
  http) module=nanovllm_voxcpm.entrypoints.openai_server ;;
  ws) module=nanovllm_voxcpm.entrypoints.ws_server ;;
  *) exec "$@" ;;
esac
shift || true
args=("$@")
has() {
  local a
  for a in "${args[@]}"; do [[ $a == "$1" || $a == "$1="* ]] && return 0; done
  return 1
}
has --host || args=(--host 0.0.0.0 "${args[@]}")
has --port || args=(--port "${PORT:-8092}" "${args[@]}")
exec python -m "$module" "${args[@]}"
